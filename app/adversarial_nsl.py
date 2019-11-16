#!/usr/bin/env python
import argparse
import json
import logging
import os

import neural_structured_learning as nsl
import tensorflow as tf
from pylib import dataset_for_split, load_training, load_validation, create_model, \
    SEED, set_log_level, TUNE_CONF
from pylib import f1_score, iou_score


def convert_to_dictionaries(image, mask):
    # mask = tf.reshape(tf.cast(mask, tf.float32), [-1])
    mask = tf.cast(mask, tf.float32)
    return {'input': image, 'label': mask}


def print__(val):
    # mask = tf.reshape(tf.cast(mask, tf.float32), [-1])
    print(type(val['label']))
    return val


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", help="Base path of data source input", type=str, required=True)
    parser.add_argument("--output", help="Base path of dataset output", type=str, required=True)

    parser.add_argument("--seed", help="Size of square image", type=int, default=SEED)
    parser.add_argument('--buffer_size', help='Shuffle buffer size', type=int, default=1000)

    # TODO: hyperparameter
    parser.add_argument('--model_arch', help='Model architecture ', type=str, default='VGG19')
    parser.add_argument('--learning_rate', help='Learning rate of optimizer (adam)', type=int, default=1e-2)
    parser.add_argument('--batch_size', help='Batch Size', type=int, default=64)
    parser.add_argument('--epochs', help='Number of epochs', type=int, default=6)
    parser.add_argument('--steps_per_epoch', help='Number of epochs', type=int, default=80)
    parser.add_argument('--hyperparam_fn_path', help='Path to json file containing hyperparameters', type=str,
                        default=None)

    parser.add_argument('--adv_step_size', type=float, default=0.2, help='The magnitude of adversarial perturbation')
    parser.add_argument('--adv_multiplier', type=float, default=0.2,
                        help='The weight of adversarial loss in the training objective, relative to the labeled loss.')
    parser.add_argument('--adv_grad_norm', type=str, default='infinity',
                        help=' The norm to measure the magnitude of adversarial perturbation.')

    parser.add_argument('--checkpoint_path', help='random seed used across', type=str, default='/pfs/out/ckpts/')
    parser.add_argument('--tensorboard_path', help='random seed used across', type=str, default='/pfs/out/logs/')

    parser.add_argument("--log", help="sets the logging level", type=str, default=os.getenv("LOG_LEVEL", "info"),
                        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "critical", "error", "warning",
                                 "info", "debug"])
    FLAGS = parser.parse_args()
    set_log_level(FLAGS.log.upper())
    tf.random.set_seed(FLAGS.seed)

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

    training_slice = dataset_for_split(FLAGS.input, "training")
    validation_slice = dataset_for_split(FLAGS.input, "validation")

    training_dataset = training_slice.map(lambda x, y: load_training(x, y, FLAGS.seed),
                                          num_parallel_calls=TUNE_CONF). \
        map(convert_to_dictionaries). \
        map(print__). \
        cache(). \
        shuffle(FLAGS.buffer_size, seed=FLAGS.seed). \
        batch(FLAGS.batch_size). \
        repeat()
    validation_dataset = validation_slice.map(load_validation,
                                              num_parallel_calls=TUNE_CONF).batch(
        FLAGS.batch_size).map(convert_to_dictionaries)

    base_model = create_model(model_arch=FLAGS.model_arch, seed=FLAGS.seed)
    adv_config = nsl.configs.make_adv_reg_config(
        multiplier=FLAGS.adv_multiplier,
        adv_step_size=FLAGS.adv_step_size,
        adv_grad_norm=FLAGS.adv_grad_norm
    )
    model = nsl.keras.AdversarialRegularization(
        base_model,
        adv_config=adv_config
    )
    adam = tf.keras.optimizers.Adam(learning_rate=FLAGS.learning_rate)
    loss = tf.keras.losses.binary_crossentropy  # BCEJaccardLoss()
    model.compile(optimizer=adam, loss=loss, metrics=[iou_score, f1_score, "binary_accuracy"])

    # TF Callbacks
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=FLAGS.tensorboard_path, histogram_freq=1)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(FLAGS.checkpoint_path + "/weights.{epoch:02d}.hdf5",
                                                     monitor="val_loss", mode="min", save_best_only=True,
                                                     verbose=1)
    lr_ctrl_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', mode="min",
                                                            factor=0.2, patience=5, min_lr=0.001,
                                                            verbose=1)
    es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode="min", patience=3,
                                                   verbose=1)

    model_history = model.fit(training_dataset, epochs=FLAGS.epochs,
                              steps_per_epoch=FLAGS.steps_per_epoch,
                              validation_data=validation_dataset)
                              # callbacks=[cp_callback, tensorboard_callback, lr_ctrl_callback, es_callback])

    model.summary()
    model.save(os.path.join(FLAGS.output, 'nsl-adverse-model.h5'))
