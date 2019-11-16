#!/usr/bin/env python

import argparse
import os

import tensorflow as tf
from pylib import bce_jaccard_loss, f1_score, iou_score, TUNE_CONF
from pylib import dataset_for_split, load_training, load_validation, SEED, set_log_level


@tf.function
def load_training_adversarial(img_fn, mask_fn, adverse_model, augmentation_seed=SEED, epsilon=0.01):
    input_image, input_mask = load_training(img_fn, mask_fn, augmentation_seed=augmentation_seed)
    # if tf.random.uniform(()) > 0.5:
    #     input_image, input_mask = create_adversarial_pattern(input_image, input_mask, adverse_model=adverse_model,
    #                                                          epsilon=epsilon)

    input_image, input_mask = create_adversarial_pattern(input_image, input_mask, adverse_model=adverse_model,
                                                         epsilon=epsilon)
    return input_image, input_mask


@tf.function
def load_validation_adversarial(img_fn, mask_fn, adverse_model, epsilon=0.01):
    input_image, input_mask = load_validation(img_fn, mask_fn)
    # if tf.random.uniform(()) > 0.5:
    #     input_image, input_mask = create_adversarial_pattern(input_image, input_mask, adverse_model=adverse_model,
    #                                                          epsilon=epsilon)

    input_image, input_mask = create_adversarial_pattern(input_image, input_mask, adverse_model=adverse_model,
                                                         epsilon=epsilon)

    return input_image, input_mask


def create_adversarial_pattern(input_image, input_mask, adverse_model=None, epsilon=0.01):
    with tf.GradientTape() as tape:
        tape.watch(input_image)
        b_input_image = tf.expand_dims(input_image, 0)
        b_predicted_mask = adverse_model(b_input_image)
        predicted_mask = b_predicted_mask[0]
        loss = bce_jaccard_loss(input_mask, predicted_mask)
    # Get the gradients of the loss w.r.t to the input image.
    gradient = tape.gradient(loss, input_image)
    # Get the sign of the gradients to create the perturbation
    perturbations = tf.sign(gradient)
    adversarial_input = tf.math.add(input_image, epsilon * perturbations)
    adversarial_input = tf.clip_by_value(adversarial_input, 0, 1)
    return adversarial_input, input_mask


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", help="Base path of data source input", type=str, required=True)
    parser.add_argument("--output", help="Base path of dataset output", type=str, required=True)

    parser.add_argument("--seed", help="Size of square image", type=int, default=SEED)
    parser.add_argument('--buffer_size', help='Shuffle buffer size', type=int, default=1000)

    parser.add_argument("--model_weights", help="Pathto model weights", type=str, required=True)
    # TODO: hyperparameter
    parser.add_argument('--eps', help='Total epsilon for FGM and PGD attacks.', type=float, default=0.05)
    parser.add_argument('--learning_rate', help='Learning rate of optimizer (adam)', type=int, default=1e-2)
    parser.add_argument('--batch_size', help='Batch Size', type=int, default=64)
    parser.add_argument('--epochs', help='Number of epochs', type=int, default=6)
    parser.add_argument('--steps_per_epoch', help='Number of epochs', type=int, default=80)
    parser.add_argument('--hyperparam_fn_path', help='Path to json file containing hyperparameters', type=str,
                        default=None)

    parser.add_argument('--checkpoint_path', help='random seed used across', type=str, default='/pfs/out/ckpts/')
    parser.add_argument('--tensorboard_path', help='random seed used across', type=str, default='/pfs/out/logs/')

    parser.add_argument("--log", help="sets the logging level", type=str, default=os.getenv("LOG_LEVEL", "info"),
                        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "critical", "error", "warning",
                                 "info", "debug"])
    FLAGS = parser.parse_args()
    set_log_level(FLAGS.log.upper())
    tf.random.set_seed(FLAGS.seed)

    model = tf.keras.models.load_model(FLAGS.model_weights, custom_objects={"bce_jaccard_loss": bce_jaccard_loss,
                                                                            "iou_score": iou_score,
                                                                            "f1_score": f1_score})

    adverserial_gen = model
    model.trainable = True
    fine_tune_at = 2
    # Freeze all the layers before the `fine_tune_at` layer
    for layer in model.layers[:fine_tune_at]:
        layer.trainable = False

    training_slice = dataset_for_split(FLAGS.input, "training")
    validation_slice = dataset_for_split(FLAGS.input, "validation")

    training_dataset = training_slice.map(
        lambda x, y: load_training_adversarial(x, y, adverserial_gen, FLAGS.seed, FLAGS.eps),
        num_parallel_calls=TUNE_CONF). \
        cache(). \
        shuffle(FLAGS.buffer_size, seed=FLAGS.seed). \
        batch(FLAGS.batch_size). \
        repeat()

    validation_dataset = validation_slice.map(
        lambda x, y: load_validation_adversarial(x, y, adverserial_gen, FLAGS.eps),
        num_parallel_calls=TUNE_CONF).batch(FLAGS.batch_size)

    adam = tf.keras.optimizers.Adam(learning_rate=FLAGS.learning_rate)
    model.compile(optimizer=adam, loss=bce_jaccard_loss, metrics=[iou_score, f1_score, "binary_accuracy"])

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
                              validation_data=validation_dataset,
                              callbacks=[cp_callback, tensorboard_callback, lr_ctrl_callback, es_callback])

    model.summary()
    model.save(os.path.join(FLAGS.output, 'adversred-model.h5'))
