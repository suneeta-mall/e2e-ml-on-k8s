import logging
import os
from typing import Iterable, List, Dict, Union

import numpy as np
import tensorflow as tf
from joblib import load
from pylib import IMG_CHANNEL, IMG_HEIGHT, IMG_WIDTH
from pylib import bce_jaccard_loss, iou_score, f1_score
from pylib import hash_file, normalize, FSClient


# see https://docs.seldon.io/projects/seldon-core/en/latest/python/python_component.html
class PetSetModel:
    def __init__(self, model_weights=os.getenv("MODEL_WEIGHTS", "model.h5"),
                 calibration_weights=os.getenv("CALIB_WEIGHTS", "calibration.weights"),
                 model_version=(os.getenv("MODEL_VERSION", "NA")),
                 model_db=os.getenv("MODEL_DB", "model_db"),
                 model_db_host=os.getenv("MODEL_STORE_ADDRESS", "localhost")):
        self._model_store = FSClient(model_db, host=model_db_host)

        if not os.path.exists(model_weights):
            self._model_store.download_file(model_version, model_weights, model_weights)
        if not os.path.exists(calibration_weights):
            self._model_store.download_file(model_version, calibration_weights, calibration_weights)

        self._model = tf.keras.models.load_model(model_weights, custom_objects={"bce_jaccard_loss": bce_jaccard_loss,
                                                                                "iou_score": iou_score,
                                                                                "f1_score": f1_score})
        self._calibration_model = load(calibration_weights)

        model_sha = hash_file(model_weights)
        calib_sha = hash_file(calibration_weights)
        self.meta_dict = {"model_version": model_version,
                          "model_sha": model_sha,
                          "calibration_sha": calib_sha}

    def class_names(self) -> Iterable[str]:
        logging.info("class name  is called")
        return ["Pets", "Background", "Border"]

    def transform_input(self, x: np.ndarray, names: Iterable[str], meta: Dict = None) -> Union[
        np.ndarray, List, str, bytes]:
        # Single image not batched tensor
        logging.info("transform input is called %s ", x.shape)
        return PetSetModel.__reshape_tensor_if_needed(x)

    @staticmethod
    def __reshape_tensor_if_needed(x: np.ndarray) -> Union[np.ndarray, List, str, bytes]:
        _input = x
        # Add batch axis if its only image tensor
        if len(x.shape) == 3:
            _input = tf.expand_dims(x, 0)
        # if input is noot normalized
        if np.max(_input) > 1:
            _input, _ = normalize(_input, None)
        return _input

    def _apply_calibration(self, x: np.ndarray) -> Union[np.ndarray, List, str, bytes]:
        logging.info("Calibration is applied %s ", x.shape)
        _batch = x.shape[0]
        flat_pmask = np.reshape(x, [-1])
        flat_cmask = self._calibration_model.predict(flat_pmask)
        cmask = np.reshape(flat_cmask, [_batch, IMG_WIDTH, IMG_HEIGHT, IMG_CHANNEL])
        return cmask[0] if x.shape[0] == 1 else cmask

    def predict(self, x, names: Iterable[str], meta: Dict = None):
        logging.info("Predict called on tensor %s ", x.shape)
        x = PetSetModel.__reshape_tensor_if_needed(x)
        _preds = self._model.predict(x)
        _preds = self._apply_calibration(_preds)
        _preds = self.transform_output(_preds, names=[])
        return _preds

    def transform_output(self, x: np.ndarray, names: Iterable[str], meta: Dict = None) -> \
            Union[np.ndarray, List, str, bytes]:
        logging.info("Transform_output  is called %s ", x.shape)
        res = x[0] if x.shape[0] == 1 else x
        return res

    def tags(self) -> Dict:
        return self.meta_dict

    def send_feedback(self, x, names: Iterable[str], reward, truth, routing=None):
        # https://github.com/SeldonIO/seldon-core/tree/master/examples/models/template_model_with_feedback
        logging.info("Send feedback called")
        return []

    def metrics(self):
        return [
            {"type": "COUNTER", "key": "mycounter", "value": 1},  # a counter which will increase by the given value
            {"type": "GAUGE", "key": "mygauge", "value": 100},  # a gauge which will be set to given value
            {"type": "TIMER", "key": "mytimer", "value": 20.2},
            # a timer which will add sum and count metrics - assumed millisecs
        ]
