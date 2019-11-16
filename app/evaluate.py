#!/usr/bin/env python

import argparse
import json
import logging
import os

import pandas as pd
import tensorflow as tf
from joblib import load
from pylib import bce_jaccard_loss, iou_score, f1_score, set_log_level, accuracy, mean_iou, precision
from pylib import load_validation as load_test, dataset_for_split, calibrated_mask, display, binarize, \
    TUNE_CONF, set_global_determinism, SEED

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_weights", help="Pathto model weights", type=str, required=True)
    parser.add_argument("--calibration_weights", help="Path to calibration weights", type=str, required=True)
    parser.add_argument("--hyperparameters", help="Base path of hyperparameters file", type=str)

    parser.add_argument("--input", help="Base path of dataset output", type=str, required=True)
    parser.add_argument("--output", help="Base path of dataset output", type=str, required=True)
    parser.add_argument('--batch_size', help='Batch Size', type=int, default=64)
    parser.add_argument("--seed", help="Size of square image", type=int, default=SEED)

    tip = "Whether to achieve full or close reproducibility"
    parser.add_argument("--fully-reproducible", help=tip, dest='reproducible', action='store_true')
    parser.add_argument('--closely-reproducible', help=tip, dest='reproducible', action='store_false')
    parser.set_defaults(reproducible=True)

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

    batch_size = FLAGS.batch_size
    if FLAGS.hyperparameters and os.path.exists(FLAGS.hyperparameters):
        with open(FLAGS.hyperparameters) as f:
            hp = json.load(f)
            batch_size = hp['batch_size']
            logging.warning(f"Found hyperparameter file, loading parameters from it. Overriding batch_size.... "
                            f"\nNew batch_size is: {batch_size}")

    test_slice = dataset_for_split(FLAGS.input, "test")
    test_dataset = test_slice.map(load_test, num_parallel_calls=TUNE_CONF).batch(batch_size)

    model = tf.keras.models.load_model(FLAGS.model_weights, custom_objects={"bce_jaccard_loss": bce_jaccard_loss,
                                                                            "iou_score": iou_score,
                                                                            "f1_score": f1_score})
    calibration_model = load(FLAGS.calibration_weights)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=FLAGS.output, histogram_freq=1)

    [loss_score, iou_score_score, f1_score_score, accuracy_score] = model.evaluate(test_dataset,
                                                                                   callbacks=[tensorboard_callback])
    truth, pred_mask, calib_mask = None, None, None
    loop = ""
    for img, mask in test_dataset:
        _calib_mask, _pred_mask = calibrated_mask(model, calibration_model, img)
        if loop == "":
            calib_mask = _calib_mask
            pred_mask = _pred_mask
            truth = mask
            loop = "done"
        else:
            calib_mask = tf.concat([calib_mask, _calib_mask], axis=0)
            pred_mask = tf.concat([pred_mask, _pred_mask], axis=0)
            truth = tf.concat([truth, mask], axis=0)

    original_meanIOU = mean_iou(truth, pred_mask)
    calibrated_meanIOU = mean_iou(truth, calib_mask)

    original_iou_score = iou_score(truth, pred_mask)
    calibrated_iou_score = iou_score(truth, calib_mask)

    original_f1_score = f1_score(truth, pred_mask)
    calibrated_f1_score = f1_score(truth, calib_mask)

    original_accuracy = accuracy(truth, pred_mask)
    calibrated_accuracy = accuracy(truth, calib_mask)

    original_precision = precision(truth, pred_mask)
    calibrated_precision = precision(truth, calib_mask)

    original_bce_jaccard_loss = bce_jaccard_loss(truth, pred_mask)
    calibrated_bce_jaccard_loss = bce_jaccard_loss(truth, calib_mask)

    outcome = {
        "IOU score": ["IOU Score", original_iou_score.numpy(), calibrated_iou_score.numpy()],
        "F1 Score": ["F1 Score", original_f1_score.numpy(), calibrated_f1_score.numpy()],
        "MeanIOU": ["MeanIOU", original_meanIOU, calibrated_meanIOU],
        "Accuracy": ["Accuracy", original_accuracy, calibrated_accuracy],
        "Precision": ["Precision", original_precision, calibrated_precision],
        "Jaccard Loss": ["Jaccard Loss", original_bce_jaccard_loss.numpy(), calibrated_bce_jaccard_loss.numpy()]
    }

    df = pd.DataFrame.from_dict(outcome, orient='index', columns=["Metric Name", "Original Model", "Calibrated Model"])
    df.to_csv(os.path.join(FLAGS.output, "evaluation_result.csv"))

    print("\n\nEvaluation outcome is as following:\n\n")
    print(df.head(10), "\n\n")

    for img, mask in test_dataset.take(7):
        calib_mask, pred_mask = calibrated_mask(model, calibration_model, img)
        display([img[0], mask[0], pred_mask[0], calib_mask[0], binarize(calib_mask[0])],
                title=['Input Image', 'True Mask', 'Predicted Mask', 'Calibrated Mask', 'Thresholded Calibrated Mask'])
