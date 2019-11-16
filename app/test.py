#!/usr/bin/env python

import argparse
import json
import logging
import os
import requests

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from pylib import NumpyEncoder, normalize, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNEL, set_log_level

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="Base path of dataset output", type=str, required=True)
    parser.add_argument("--output", help="Base path of dataset output", type=str, required=True)
    parser.add_argument("--server_addr", help="Base path of dataset output", type=str,
                        default='http://localhost:5000/predict')
    parser.add_argument("--log", help="sets the logging level", type=str, default=os.getenv("LOG_LEVEL", "info"),
                        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "critical", "error", "warning",
                                 "info", "debug"])
    FLAGS = parser.parse_args()
    set_log_level(FLAGS.log.upper())

    image = tf.io.read_file(FLAGS.input)
    image = tf.image.decode_jpeg(image, channels=3)
    image, _ = normalize(image, None)

    seldon_payload = {"data": {"ndarray": image}}
    seldon_payload = json.dumps(seldon_payload, cls=NumpyEncoder)
    headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
    r = requests.post(FLAGS.server_addr, data=seldon_payload, headers=headers)
    logging.info(f"Query from prediction service was {r.ok} and status code {r.status_code}")
    if r.ok:
        prediction_payload = r.json()
        preds = prediction_payload['data']['ndarray']
        preds = np.reshape(preds, [IMG_HEIGHT, IMG_WIDTH, IMG_CHANNEL])

        joint_img = tf.keras.preprocessing.image.array_to_img(preds)
        plt.imsave(FLAGS.output, joint_img)
