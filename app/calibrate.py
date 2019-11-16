#!/usr/bin/env python

import argparse
import json
import logging
import os

import numpy as np
import tensorflow as tf
from joblib import dump
from pylib import NumpyEncoder, IMG_WIDTH, IMG_HEIGHT, IMG_CHANNEL, TUNE_CONF, SEED
from pylib import bce_jaccard_loss, iou_score, f1_score
from pylib import dataset_for_split, plot_calibration_curve, set_log_level, hash_file
from pylib import load_validation as load_test, set_global_determinism
from sklearn.isotonic import IsotonicRegression

CALIBRATION_BATCH_SIZE = 500


def flatten_tensor(tensor: tf.Tensor) -> np.ndarray:
    flat_tensor = tf.reshape(tensor, [-1])
    flat_tensor = tf.cast(flat_tensor, tf.float32)
    flat_tensor_arr = flat_tensor.numpy()
    return flat_tensor_arr


def calculate_metrics(truth, predicted, calibrated):
    f1_o = f1_score(truth, predicted)
    f1_c = f1_score(truth, calibrated)
    logging.info("F1 Score of original model is %f", f1_o)
    logging.info("F1 Score of calibrated model is %f", f1_c)

    iou_score_o = iou_score(truth, predicted)
    iou_score_c = iou_score(truth, calibrated)
    logging.info("IOU of original model is %f", iou_score_o)
    logging.info("IOU of calibrated model is %f", iou_score_c)

    loss_o = iou_score(truth, predicted)
    loss_c = iou_score(truth, calibrated)
    logging.info("Binary Cross Entropy Jaccard Loss of original model is %f", loss_o)
    logging.info("Binary Cross Entropy Jaccard Loss of calibrated model is %f", loss_c)

    calibration_outcome = {
        "original_f1_score": f1_o,
        "calibrated_f1_score": f1_c,
        "original_mean_iou": iou_score_o,
        "calibrated_mean_iou": iou_score_c,
        "original_bce_jaccard_loss": loss_o,
        "calibrated_bce_jaccard_loss": loss_c,
    }
    return calibration_outcome


def infer_versions(model_fn, calib_fn):
    version_info = {
        "model_sha": hash_file(model_fn),
        "calibration_model_sha": hash_file(calib_fn),
        # see https://docs.pachyderm.io/en/latest/reference/pipeline_spec.html#environment-variables
        "model_version": os.getenv("model_COMMIT", "NA"),
        "calibration_model_version": os.getenv("PACH_OUTPUT_COMMIT_ID", "NA"),
    }
    return version_info


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_weight", help="Base path of data source input", type=str, required=True)
    parser.add_argument("--input", help="Base path of dataset output", type=str, required=True)
    parser.add_argument("--output", help="Base path of dataset output", type=str, required=True)

    tip = "Whether to achieve full or close reproducibility"
    parser.add_argument("--fully-reproducible", help=tip, dest='reproducible', action='store_true')
    parser.add_argument('--closely-reproducible', help=tip, dest='reproducible', action='store_false')
    parser.set_defaults(reproducible=True)

    tip = "Whether to explicitly download files"
    parser.add_argument("--force-download", help=tip, dest='download', action='store_true')
    parser.add_argument('--no-download', help=tip, dest='download', action='store_false')
    parser.set_defaults(download=False)

    parser.add_argument("--seed", help="Size of square image", type=int, default=SEED)
    parser.add_argument('--buffer_size', help='Shuffle buffer size', type=int, default=1000)
    parser.add_argument('--batch_size', help='Batch Size', type=int, default=64)

    parser.add_argument("--log", help="sets the logging level", type=str, default=os.getenv("LOG_LEVEL", "info"),
                        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "critical", "error", "warning",
                                 "info", "debug"])

    FLAGS = parser.parse_args()
    set_log_level(FLAGS.log.upper())
    set_global_determinism(seed=FLAGS.seed, fast_n_close=not FLAGS.reproducible)

    calibration_slice = dataset_for_split(FLAGS.input, "calibration")
    calibration_dataset = calibration_slice.map(load_test,
                                                num_parallel_calls=TUNE_CONF).batch(
        CALIBRATION_BATCH_SIZE)

    model = tf.keras.models.load_model(FLAGS.model_weight, custom_objects={"bce_jaccard_loss": bce_jaccard_loss,
                                                                           "iou_score": iou_score,
                                                                           "f1_score": f1_score})
    test_batch_size = 0
    for idx, (img, mask) in calibration_dataset.enumerate():
        # Training set
        if idx == 0:
            img_train, truth_mask_train, predicted_mask_train = img, mask, model.predict(img)
        elif idx == 1:
            # Test
            img_test, truth_mask_test, predicted_mask_test = img, mask, model.predict(img)
            test_batch_size = tf.shape(img).numpy()[0]
        else:
            logging.warning("Skipping some data!!!")
            break

    # Flatten the base model predictions and true values
    predicted_mask_train_arr = flatten_tensor(predicted_mask_train)
    truth_mask_train_arr = flatten_tensor(truth_mask_train)

    predicted_mask_test_arr = flatten_tensor(predicted_mask_test)
    truth_mask_test_arr = flatten_tensor(truth_mask_test)

    iso_regression = IsotonicRegression(out_of_bounds='clip')
    iso_regression.fit(predicted_mask_train_arr, truth_mask_train_arr)
    p_calibrated = iso_regression.predict(predicted_mask_test_arr)
    calibration_model_fn = os.path.join(FLAGS.output, 'calibration.weights')
    dump(iso_regression, calibration_model_fn)

    plot_calibration_curve(truth_mask_test_arr, p_calibrated, output=FLAGS.output)

    # Convert 1-d ndarray to tensor of 3 channel
    p_calibrated = np.reshape(p_calibrated, [test_batch_size, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNEL])

    metrics = calculate_metrics(truth_mask_test, predicted_mask_test, p_calibrated)

    metadata = {
        "version": infer_versions(FLAGS.model_weight, calibration_model_fn),
        "metrics": metrics,
    }

    with open(os.path.join(FLAGS.output, 'calibration_outcome.json'), 'w') as f:
        json.dump(metadata, f, cls=NumpyEncoder)
