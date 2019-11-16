#!/usr/bin/env python
import argparse
import hashlib
import os
import shutil

import numpy as np
import tensorflow as tf
from PIL import Image

from pylib import IMG_CHANNEL, IMG_WIDTH, set_log_level


def load_and_preprocess_mask(path: str, size: int = IMG_WIDTH):
    image = tf.io.read_file(path)
    image = tf.image.decode_png(image, channels=IMG_CHANNEL)
    image = tf.image.resize(image, [size, size])
    return image


def load_and_preprocess_image(path: str, size: int = IMG_WIDTH):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=IMG_CHANNEL)
    image = tf.image.resize(image, [size, size])
    # TODO: Creating really blue image .. check why
    # otherwise combine
    # image = tf.image.per_image_standardization(image)
    return image


def convert_multichannel_img(data: np.ndarray):
    data = data[:, :, 0]
    h, w = data.shape
    multic = np.zeros([h, w, IMG_CHANNEL])
    # 1 is animal/foreground, 2 is background and 3 is unclassified/unsure/border
    # multic[:, :, 0] = np.where(data == 1, 1, 0)
    multic[:, :, 0] = np.where(data == 1, 1, 0)
    multic[:, :, 1] = np.where(data == 2, 1, 0)
    multic[:, :, 2] = np.where(data == 3, 1, 0)
    return multic


def save_tensor_as_img(img_file: str, tensor: tf.Tensor):
    if isinstance(tensor, tf.Tensor):
        data = tensor.numpy()
    else:
        data = tensor
    im = Image.fromarray(data.astype('uint8'))
    os.makedirs(os.path.dirname(img_file), exist_ok=True)
    im.save(img_file)


def split_by_partition(idx: int):
    if idx in range(0, 7):
        return "training"
    elif idx == 7:
        return "validation"
    elif idx == 8:
        return "test"
    elif idx == 9:
        return "calibration"
    else:
        raise ValueError(f"Partition id can only be b/w 0-9. Provided {idx}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", help="Base path of data source input", type=str, default="/pfs/warehoue")
    parser.add_argument("--output", help="Base path of dataset output", type=str, default="/pfs/out")

    parser.add_argument("--img_size", help="Size of square image", type=int, default=IMG_WIDTH)
    parser.add_argument("--log", help="sets the logging level", type=str, default=os.getenv("LOG_LEVEL", "info"),
                        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "critical", "error", "warning",
                                 "info", "debug"])
    FLAGS = parser.parse_args()

    set_log_level(FLAGS.log.upper())

    for _dir in os.scandir(FLAGS.input):
        if not _dir.is_dir():
            continue
        petset_id = os.path.basename(_dir)

        partition_idx = int(hashlib.md5((os.path.basename(petset_id)).encode()).hexdigest(), 16) % 10
        split_name = split_by_partition(partition_idx)

        img_fn = os.path.join(_dir.path, "image.jpg")
        mask_fn = os.path.join(_dir.path, "mask.png")
        meta_fn = os.path.join(_dir.path, "metadata.json")

        if not os.path.exists(img_fn) or not os.path.exists(mask_fn):
            raise FileNotFoundError(f"Expected both image and mask to be found for petset data {_dir}")

        img_tensor = load_and_preprocess_image(img_fn)
        mask_tensor = load_and_preprocess_mask(mask_fn)

        mc_mask_tensor = convert_multichannel_img(mask_tensor)

        img_out_fn = os.path.join(FLAGS.output, split_name, petset_id, "image.jpg")
        mask_out_fn = os.path.join(FLAGS.output, split_name, petset_id, "mask.png")
        meta_out_fn = os.path.join(FLAGS.output, split_name, petset_id, "metadata.json")

        save_tensor_as_img(mask_out_fn, mc_mask_tensor)
        save_tensor_as_img(img_out_fn, img_tensor)
        shutil.copy(meta_fn, meta_out_fn)
