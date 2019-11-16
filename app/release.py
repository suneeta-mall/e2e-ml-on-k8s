#!/usr/bin/env python

import argparse
import logging
import os

import pandas as pd
from pylib import set_log_level, SeldonController

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="Base path of summary csv", type=str, required=True)
    parser.add_argument('--container_version', help='Version of container to run for tunning', type=str, default=None)

    parser.add_argument("--version", help="Version to tag model with", type=str,
                        default=os.getenv("evaluate_COMMIT", "latest"))
    parser.add_argument("--model_db", help="Name og  model db", type=str, default="evaluate")

    parser.add_argument("--log", help="sets the logging level", type=str, default=os.getenv("LOG_LEVEL", "info"),
                        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "critical", "error", "warning",
                                 "info", "debug"])
    FLAGS = parser.parse_args()
    set_log_level(FLAGS.log.upper())

    if not FLAGS.container_version:
        with open("/app/version") as f:
            FLAGS.container_version = f.read()

    df = pd.read_csv(FLAGS.input)
    if df.loc[df['Metric Name'] == "Accuracy"]['Calibrated Model'].values[0] > 0.92:
        logging.info("Model passed quality check!")
        sc = SeldonController(FLAGS.container_version, FLAGS.model_db, FLAGS.version)
        sc.create_serving()
    else:
        logging.error("This version of model failed quality check!")
        raise Exception("Boo!")
