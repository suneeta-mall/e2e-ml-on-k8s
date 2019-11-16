#!/usr/bin/env python3

import argparse
import os

import tensorflow as tf
from pylib import bce_jaccard_loss, f1_score, iou_score, TUNE_CONF
from pylib import dataset_for_split, load_validation as load_test, set_log_level, binarize, display

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", help="Base path of data source input", type=str, required=True)
    parser.add_argument("--output", help="Base path of dataset output", type=str, required=True)

    parser.add_argument("--model_weights", help="Pathto model weights", type=str, required=True)
    parser.add_argument('--batch_size', help='Batch Size', type=int, default=64)
    parser.add_argument('--eps', help='Total epsilon for FGM and PGD attacks.', type=float, default=0.05)
    parser.add_argument("--log", help="sets the logging level", type=str, default=os.getenv("LOG_LEVEL", "info"),
                        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "critical", "error", "warning",
                                 "info", "debug"])
    FLAGS = parser.parse_args()
    set_log_level(FLAGS.log.upper())

    test_slice = dataset_for_split(FLAGS.input, "calibration")
    test_dataset = test_slice.map(load_test, num_parallel_calls=TUNE_CONF).batch(FLAGS.batch_size)

    model = tf.keras.models.load_model(FLAGS.model_weights, custom_objects={"bce_jaccard_loss": bce_jaccard_loss,
                                                                            "iou_score": iou_score,
                                                                            "f1_score": f1_score})


    def create_adversarial_pattern(input_image, input_mask):
        with tf.GradientTape() as tape:
            tape.watch(input_image)
            predicted_mask = model(input_image)
            loss = bce_jaccard_loss(input_mask, predicted_mask)
        # Get the gradients of the loss w.r.t to the input image.
        gradient = tape.gradient(loss, input_image)
        # Get the sign of the gradients to create the perturbation
        signed_grad = tf.sign(gradient)
        return signed_grad, predicted_mask


    for image, mask in test_dataset.take(7):
        perturbations, predicted_mask = create_adversarial_pattern(image, mask)
        adv_x = image + (FLAGS.eps * perturbations)
        adv_x = tf.clip_by_value(adv_x, 0, 1)
        adv_pred = model.predict(adv_x)
        display([image[0], mask[0], binarize(predicted_mask[0]), perturbations[0], adv_x[0],
                 binarize(adv_pred[0])],
                title=['Input Image', 'True Mask', 'Predicted Mask', 'Perturbations', 'Adverserial Mask',
                       'Adverserial Prediction Mask'])
