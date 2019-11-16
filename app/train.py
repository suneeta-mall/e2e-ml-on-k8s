#!/usr/bin/env python
import argparse
import json
import logging
import os

from pylib import SEED, set_log_level, set_global_determinism
from pylib import download_input, train_model


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

    model, model_history = train_model(FLAGS.input, FLAGS.model_arch, FLAGS.batch_size, FLAGS.epochs,
                                       FLAGS.learning_rate, FLAGS.steps_per_epoch,
                                       FLAGS.seed, FLAGS.buffer_size,
                                       FLAGS.tensorboard_path, FLAGS.checkpoint_path)
    model.save(os.path.join(FLAGS.output, 'model.h5'))
