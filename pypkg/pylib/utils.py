import hashlib
import json
import random
import os
import logging
import subprocess

import numpy as np
import tensorflow as tf

BUFFER_SIZE = 65536
SEED = 42


def set_global_determinism(seed=SEED, fast_n_close=False):
    """
        Enable 100% reproducibility on operations related to tensor and randomness.
        Parameters:
        seed (int): seed value for global randomness
        fast_n_close (bool): whether to achieve efficient at the cost of determinism/reproducibility
    """
    set_seeds(seed=seed)
    if fast_n_close:
        return

    logging.warning("*******************************************************************************")
    logging.warning("*** set_global_determinism is called,setting full determinism, will be slow ***")
    logging.warning("*******************************************************************************")

    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    # https://www.tensorflow.org/api_docs/python/tf/config/threading/set_inter_op_parallelism_threads
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    from tfdeterminism import patch
    patch()


def set_seeds(seed=SEED):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)



# pylint: disable=method-hidden
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, tf.Tensor):
            return obj.numpy().tolist()
        return json.JSONEncoder.default(self, obj)
# pylint: enable=method-hidden


def hash_file(file: str) -> str:
    file_hash = hashlib.sha256()
    with open(file, 'rb') as f:
        fb = f.read(BUFFER_SIZE)
        while fb:
            file_hash.update(fb)
            fb = f.read(BUFFER_SIZE)
    return file_hash.hexdigest()


def set_log_level(level: str):
    _level = logging.getLevelName(level)
    logging.basicConfig(level=_level)
    logging.getLogger().setLevel(_level)


def execute_cmd(cmd_list):
    process = subprocess.run(cmd_list, check=True, stdout=subprocess.PIPE, universal_newlines=True)
    process.check_returncode()
    return process.stdout
