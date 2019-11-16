import os

import tensorflow as tf
from .utils import SEED

IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNEL = 3
OUTPUT_CHANNELS = 3


def normalize(img_fn, mask_fn):
    input_image = tf.cast(img_fn, tf.float32) / 128.0 - 1
    return input_image, mask_fn


def load_img_mask(img_fn, mask_fn):
    image = tf.io.read_file(img_fn)
    image = tf.image.decode_jpeg(image, channels=IMG_CHANNEL)
    mask = tf.io.read_file(mask_fn)
    mask = tf.image.decode_png(mask, channels=IMG_CHANNEL)
    return image, mask


@tf.function
def load_training(img_fn, mask_fn, augmentation_seed=SEED):
    input_image, input_mask = load_img_mask(img_fn, mask_fn)

    input_image = tf.image.random_flip_left_right(input_image, seed=augmentation_seed)
    input_mask = tf.image.random_flip_left_right(input_mask, seed=augmentation_seed)

    input_image, input_mask = normalize(input_image, input_mask)
    return input_image, input_mask


def load_validation(img_fn, mask_fn):
    input_image, input_mask = load_img_mask(img_fn, mask_fn)
    input_image, input_mask = normalize(input_image, input_mask)
    return input_image, input_mask


def dataset_for_split(base_dir: str, split_id: str, image_fn="*.jpg", mask_name='*.png'):
    img_paths_ds = tf_glob_dataset(os.path.join(base_dir, split_id, '**', image_fn))
    mask_paths_ds = tf_glob_dataset(os.path.join(base_dir, split_id, '**', mask_name))
    _ds = tf.data.Dataset.zip((img_paths_ds, mask_paths_ds))
    return _ds


def tf_glob_dataset(glob_pattern: str):
    _paths = tf.io.gfile.glob(glob_pattern)
    _ds = tf.data.Dataset.from_tensor_slices(_paths)
    return _ds
