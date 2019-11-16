#!/usr/bin/env python
import argparse
import json
import logging
import os

import numpy as np

from hyperopt import hp
from pylib import SEED, set_log_level, train_model, set_global_determinism
import ray
from ray import tune
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.suggest.hyperopt import HyperOptSearch


def train_model_hp(config: dict, reporter):
    model, history = train_model(FLAGS.input, config['model_arch'], batch_size=config['batch_size'],
                                 epochs=config['epochs'],
                                 learning_rate=config['learning_rate'],
                                 steps_per_epoch=config['steps_per_epoch'],
                                 seed=FLAGS.seed, buffer_size=FLAGS.buffer_size,
                                 tensorboard_path=config['rays_ckpt'], checkpoint_path=config['rays_ckpt'])
    metric = {
        "val_loss": history.history['val_loss'][-1],
        "val_iou_score": history.history['val_iou_score'][-1],
        "val_f1_score": history.history['val_f1_score'][-1],
        "val_binary_accuracy": history.history['val_binary_accuracy'][-1],
    }
    reporter(**metric)


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

    parser.add_argument('--num_samples', help='num_samples', type=int, default=1)

    parser.add_argument("--ray_logdir", help="Directory to log ray trials and checkpoints", type=str, default=None)
    parser.add_argument("--log", help="sets the logging level", type=str, default=os.getenv("LOG_LEVEL", "info"),
                        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "critical", "error", "warning",
                                 "info", "debug"])
    FLAGS = parser.parse_args()
    set_log_level(FLAGS.log.upper())
    set_global_determinism(seed=FLAGS.seed, fast_n_close=not FLAGS.reproducible)

    rays_log_dir = FLAGS.ray_logdir if FLAGS.ray_logdir else os.path.join(FLAGS.output, "rays_results")
    rays_ckpt = os.path.join(rays_log_dir, "tf-checkpoints")
    os.makedirs(rays_ckpt, exist_ok=True)

    np.random.seed(seed=FLAGS.seed)
    ray.init(logging_level=logging.WARN)

    ahb = AsyncHyperBandScheduler(
        time_attr="training_iteration",
        metric="val_loss",
        mode="min",
        grace_period=2,
        max_t=90)

    hp_search_space = {
        'model_arch': hp.choice("model_arch", ["VGG19", "MobileNetV2"]),
        'batch_size': hp.choice("batch_size", np.arange(60, 81, dtype=int)),
        'epochs': hp.choice("epochs", np.arange(6, 13, dtype=int)),
        'steps_per_epoch': hp.choice("steps_per_epoch", np.arange(80, 91, dtype=int)),
        'learning_rate': hp.uniform("learning_rate", 1e-2, 1e-3),
        'rays_ckpt': hp.choice("rays_ckpt", [rays_ckpt])
    }
    current_best_hp = [
        {
            'model_arch': 0,  # "VGG19",
            'batch_size': 5,  # 64,
            'epochs': 6,  # 12,
            'steps_per_epoch': 0,  # 80,
            'learning_rate': 1e-2,
            'rays_ckpt': 0  # rays_ckpt
        }
    ]
    search_algo = HyperOptSearch(
        hp_search_space,
        max_concurrent=4,
        metric="val_loss",
        mode="min",
        points_to_evaluate=current_best_hp,
        random_state_seed=SEED)

    analysis = tune.run(train_model_hp,
                        name="hp_trials",
                        scheduler=ahb,
                        local_dir=rays_log_dir,
                        search_alg=search_algo,
                        **{
                            "stop": {
                                "training_iteration": 1,
                                "val_loss": 0.2,
                            },
                            "num_samples": FLAGS.num_samples,
                            "config": {
                                "iterations": 100,
                            },
                            "resources_per_trial": {
                                "cpu": 1
                            },
                        })

    best_params = analysis.get_best_config(metric="val_loss", mode="min")
    logging.info("Best config is %s", best_params)
    with open(os.path.join(FLAGS.output, "optimal_hp.json"), 'w') as f:
        json.dump(best_params, f)

    df = analysis.dataframe()
    df.to_csv(os.path.join(rays_log_dir, "hp_analysis_outcome.csv"))
