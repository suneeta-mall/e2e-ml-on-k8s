#!/usr/bin/env python

import argparse
import os

import tensorflow as tf
from pylib import bce_jaccard_loss, iou_score, f1_score, TUNE_CONF
from pylib import display, dataset_for_split, load_validation as load_test, binarize, set_log_level


def show_predictions(dataset, num=1):
    _items = []
    for image, mask in dataset.take(num):
        pred_mask = model.predict(image)
        _items.append([image[0], mask[0], pred_mask[0], binarize(pred_mask[0])])
        # display([image[0], mask[0], pred_mask[0], binarize(pred_mask[0])],
        #         title=['Input Image', 'True Mask', 'Predicted Mask', 'Threshold Predicted Mask'])
    display(_items,
            title=['Input Image', 'True Mask', 'Predicted Mask', 'Threshold Predicted Mask'], one_per_window=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_weight", help="Base path of data source input", type=str, required=True)
    parser.add_argument("--input", help="Base path of dataset output", type=str, required=True)

    parser.add_argument('--buffer_size', help='Shuffle buffer size', type=int, default=1000)
    parser.add_argument('--batch_size', help='Batch Size', type=int, default=64)
    parser.add_argument('--count', help='How many examples to show', type=int, default=2)
    parser.add_argument('--set', help='How many examples to show', type=str, default="test",
                        choices=["training", "validation", "calibration", "test"])

    parser.add_argument("--log", help="sets the logging level", type=str, default=os.getenv("LOG_LEVEL", "info"),
                        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "critical", "error", "warning",
                                 "info", "debug"])

    FLAGS = parser.parse_args()

    set_log_level(FLAGS.log.upper())

    test_slice = dataset_for_split(FLAGS.input, FLAGS.set)
    test_dataset = test_slice.map(load_test,
                                  num_parallel_calls=TUNE_CONF).batch(FLAGS.batch_size)

    model = tf.keras.models.load_model(FLAGS.model_weight, custom_objects={"bce_jaccard_loss": bce_jaccard_loss,
                                                                           "iou_score": iou_score,
                                                                           "f1_score": f1_score})
    show_predictions(test_dataset, FLAGS.count)
