#!/usr/bin/env python
import argparse
import json
import logging
import os

import tensorflow as tf
from pylib import SEED, set_log_level, set_global_determinism, TUNE_CONF, IMG_CHANNEL, IMG_WIDTH, IMG_HEIGHT
from pylib import dataset_for_split, download_input, normalize

_cid_dtype = {"ClassId": tf.io.FixedLenFeature([], dtype=tf.int64, default_value=1)}


def load_img_mask(img_fn, mask_fn):
    image = tf.io.read_file(img_fn)
    image = tf.image.decode_jpeg(image, channels=IMG_CHANNEL)
    mask = tf.io.read_file(mask_fn)
    class_id = tf.io.parse_single_example(tf.io.decode_json_example(mask), _cid_dtype)['ClassId']
    return image, class_id


@tf.function
def load_image_train(img_fn, mask_fn, augmentation_seed=SEED):
    input_image, label = load_img_mask(img_fn, mask_fn)

    input_image = tf.image.random_flip_left_right(input_image, seed=augmentation_seed)

    input_image, _ = normalize(input_image, label)
    return input_image, label


def load_image_test(img_fn, mask_fn):
    input_image, label = load_img_mask(img_fn, mask_fn)
    input_image, label = normalize(input_image, label)
    return input_image, label


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", help="Base path of data source input", type=str, required=True)
    parser.add_argument("--output", help="Base path of dataset output", type=str, required=True)

    parser.add_argument("--seed", help="Size of square image", type=int, default=SEED)
    parser.add_argument('--buffer_size', help='Shuffle buffer size', type=int, default=1000)

    tip = "Whether to achieve full or close reproducibility"
    parser.add_argument("--fully-reproducible", help=tip, dest='reproducible', action='store_true')
    parser.add_argument('--closely-reproducible', help=tip, dest='reproducible', action='store_false')
    parser.set_defaults(reproducible=True)

    # TODO: hyperparameter
    parser.add_argument('--model_arch', help='Model architecture ', type=str, default='VGG19')
    parser.add_argument('--learning_rate', help='Learning rate of optimizer (adam)', type=float, default=1e-2)
    parser.add_argument('--batch_size', help='Batch Size', type=int, default=64)
    parser.add_argument('--epochs', help='Number of epochs', type=int, default=6)
    parser.add_argument('--steps_per_epoch', help='Number of epochs', type=int, default=80)
    parser.add_argument('--hyperparam_fn_path', help='Path to json file containing hyperparameters', type=str,
                        default=None)

    parser.add_argument('--checkpoint_path', help='random seed used across', type=str, default='/pfs/out/ckpts/')
    parser.add_argument('--tensorboard_path', help='random seed used across', type=str, default='/pfs/out/logs/')

    tip = "Whether to explicitly download files"
    parser.add_argument("--force-download", help=tip, dest='download', action='store_true')
    parser.add_argument('--no-download', help=tip, dest='download', action='store_false')
    parser.set_defaults(download=False)

    parser.add_argument("--log", help="sets the logging level", type=str, default=os.getenv("LOG_LEVEL", "info"),
                        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "critical", "error", "warning",
                                 "info", "debug"])
    FLAGS = parser.parse_args()
    set_log_level(FLAGS.log.upper())
    set_global_determinism(seed=FLAGS.seed, fast_n_close=not FLAGS.reproducible)

    if FLAGS.hyperparam_fn_path:
        argparse_dict = vars(FLAGS)
        with open(FLAGS.hyperparam_fn_path) as f:
            logging.warning("Found hyper-parameter file, will override hyper parameters provided via CLI")
            logging.warning("Originally provided parameters are: "
                            f"Input: {FLAGS.input}, Model: {FLAGS.model_arch}, Batch Size: {FLAGS.batch_size}, "
                            f"Epochs: {FLAGS.epochs}, Learning Rate: {FLAGS.learning_rate}, "
                            f"Steps Per Epoch: {FLAGS.steps_per_epoch}")
            hps = json.load(f)
            argparse_dict.update(hps)
            logging.warning(f"Revised parameters from hyper-parameter file {FLAGS.hyperparam_fn_path} are: \n"
                            f"Input: {FLAGS.input}, Model: {FLAGS.model_arch}, Batch Size: {FLAGS.batch_size}, "
                            f"Epochs: {FLAGS.epochs}, Learning Rate: {FLAGS.learning_rate}, "
                            f"Steps Per Epoch: {FLAGS.steps_per_epoch}")

    if FLAGS.download:
        logging.warning(f"Force download is enabled, downloading from pachd")
        download_input(FLAGS.input)

    os.makedirs(FLAGS.checkpoint_path, exist_ok=True)
    os.makedirs(FLAGS.tensorboard_path, exist_ok=True)

    training_slice = dataset_for_split(FLAGS.input, "training", image_fn="*.jpg", mask_name='*.json')
    validation_slice = dataset_for_split(FLAGS.input, "validation", image_fn="*.jpg", mask_name='*.json')
    test_slice = dataset_for_split(FLAGS.input, "test", image_fn="*.jpg", mask_name='*.json')

    training_dataset = training_slice.map(lambda x, y: load_image_train(x, y, FLAGS.seed),
                                          num_parallel_calls=TUNE_CONF). \
        cache(). \
        shuffle(FLAGS.buffer_size, seed=FLAGS.seed). \
        batch(FLAGS.batch_size). \
        repeat()

    validation_dataset = validation_slice.map(load_image_test, num_parallel_calls=TUNE_CONF).batch(FLAGS.batch_size)
    test_dataset = test_slice.map(load_image_test, num_parallel_calls=TUNE_CONF).batch(FLAGS.batch_size)

    model_app = tf.keras.applications.VGG19 if FLAGS.model_arch == "VGG19" else tf.keras.applications.MobileNetV2
    base_model = model_app(input_shape=[IMG_HEIGHT, IMG_WIDTH, IMG_CHANNEL], include_top=False)
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(38, activation='softmax')
    ])
    adam = tf.keras.optimizers.Adam(learning_rate=FLAGS.learning_rate)
    model.compile(optimizer=adam, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=FLAGS.tensorboard_path, histogram_freq=1)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(FLAGS.checkpoint_path + "/weights.{epoch:02d}.hdf5",
                                                     monitor="val_loss", mode="min", save_best_only=True,
                                                     verbose=1)
    lr_ctrl_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', mode="min",
                                                            factor=0.2, patience=5, min_lr=0.001,
                                                            verbose=1)
    es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode="min", patience=3,
                                                   verbose=1)

    history = model.fit(training_dataset, epochs=FLAGS.epochs,
                        steps_per_epoch=FLAGS.steps_per_epoch,
                        validation_data=validation_dataset,
                        callbacks=[cp_callback, tensorboard_callback, lr_ctrl_callback, es_callback])
    model.save(os.path.join(FLAGS.output, 'adv_classifier_model.h5'))
    model.evaluate(test_dataset)
