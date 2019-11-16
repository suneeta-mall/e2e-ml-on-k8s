import numpy as np
import tensorflow as tf

from .datapipeline import IMG_CHANNEL, IMG_WIDTH, IMG_HEIGHT, OUTPUT_CHANNELS
from .datapipeline import dataset_for_split, load_training, load_validation
from .metrics import bce_jaccard_loss, f1_score, iou_score, MetricCollectorCallback
from .utils import SEED, set_seeds


TUNE_CONF = tf.data.experimental.AUTOTUNE # 1


def binarize(mask: tf.Tensor) -> tf.Tensor:
    return tf.math.round(mask)


def calibrated_mask(base_model: tf.keras.Model, calibrated_model, image) -> (np.ndarray, np.ndarray):
    _batch = image.shape[0]
    pmask = base_model.predict(image)
    flat_pmask = np.reshape(pmask, [-1])
    flat_cmask = calibrated_model.predict(flat_pmask)
    cmask = np.reshape(flat_cmask, [_batch, IMG_WIDTH, IMG_HEIGHT, IMG_CHANNEL])
    return cmask, pmask


def _upsample(filters, size, seed=SEED):
    initializer = tf.random_normal_initializer(0., 0.02, seed=seed)

    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                               padding='same',
                                               kernel_initializer=initializer,
                                               use_bias=False))
    result.add(tf.keras.layers.BatchNormalization())
    result.add(tf.keras.layers.Dropout(0.5, seed=SEED))
    result.add(tf.keras.layers.ReLU())
    return result


def _unet_model(up_stack, down_stack, output_channels):
    # This is the last layer of the model
    last = tf.keras.layers.Conv2DTranspose(output_channels, 3, strides=2, padding='same', activation='sigmoid')
    inputs = tf.keras.layers.Input(shape=[IMG_HEIGHT, IMG_WIDTH, IMG_CHANNEL])
    x = inputs

    # Downsampling through the model
    skips = down_stack(x)
    x = skips[-1]
    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = tf.keras.layers.Concatenate()
        x = concat([x, skip])
    x = last(x)
    return tf.keras.Model(inputs=inputs, outputs=x)


def create_mobilenet():
    base_model = tf.keras.applications.MobileNetV2(input_shape=[IMG_HEIGHT, IMG_WIDTH,
                                                                IMG_CHANNEL], include_top=False)
    # Use the activations of these layers
    layer_names = [
        'block_1_expand_relu',  # 64x64
        'block_3_expand_relu',  # 32x32
        'block_6_expand_relu',  # 16x16
        'block_13_expand_relu',  # 8x8
        'block_16_project',  # 4x4
    ]
    layers = [base_model.get_layer(name).output for name in layer_names]
    return base_model, layers


def create_vgg16():
    base_model = tf.keras.applications.VGG19(input_shape=[IMG_HEIGHT, IMG_WIDTH,
                                                          IMG_CHANNEL], include_top=False)
    # Use the activations of these layers
    layer_names = [
        'block1_pool',  # 64x64
        'block2_pool',  # 32x32
        'block3_pool',  # 16x16
        'block4_pool',  # 8x8
        'block5_pool',  # 4x4
    ]
    layers = [base_model.get_layer(name).output for name in layer_names]
    return base_model, layers


def create_model(model_arch="MobileNetV2", seed=SEED):
    set_seeds(seed=seed)
    if model_arch == "VGG19":
        base_model, layers = create_vgg16()
    else:
        base_model, layers = create_mobilenet() # Default to model_arch == "MobileNetV2":

    # Create the feature extraction model
    down_stack = tf.keras.Model(inputs=base_model.input, outputs=layers)
    down_stack.trainable = False

    up_stack = [
        _upsample(512, 3, seed=seed),  # 4x4 -> 8x8
        _upsample(256, 3, seed=seed),  # 8x8 -> 16x16
        _upsample(128, 3, seed=seed),  # 16x16 -> 32x32
        _upsample(64, 3, seed=seed),  # 32x32 -> 64x64
    ]

    model = _unet_model(up_stack, down_stack, OUTPUT_CHANNELS)
    return model


def train_model(dataset_loc, model_arch, batch_size=64, epochs=1, learning_rate=1e-2, steps_per_epoch=80,
                seed=SEED, buffer_size=1000, tensorboard_path='.', checkpoint_path='.'):
    training_slice = dataset_for_split(dataset_loc, "training")
    validation_slice = dataset_for_split(dataset_loc, "validation")

    training_dataset = training_slice.map(lambda x, y: load_training(x, y, seed),
                                          num_parallel_calls=TUNE_CONF). \
        cache(). \
        shuffle(buffer_size, seed=seed). \
        batch(batch_size). \
        repeat()
    validation_dataset = validation_slice.map(load_validation,
                                              num_parallel_calls=TUNE_CONF).batch(batch_size)

    _model = create_model(model_arch=model_arch, seed=seed)
    adam = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    _model.compile(optimizer=adam, loss=bce_jaccard_loss, metrics=[iou_score, f1_score, "binary_accuracy"])

    # TF Callbacks
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_path, histogram_freq=1)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path + "/weights.{epoch:02d}.hdf5",
                                                     monitor="val_loss", mode="min", save_best_only=True,
                                                     verbose=1)
    lr_ctrl_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', mode="min",
                                                            factor=0.2, patience=5, min_lr=0.001,
                                                            verbose=1)
    es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode="min", patience=3,
                                                   verbose=1)

    _model_history = _model.fit(training_dataset, epochs=epochs,
                                steps_per_epoch=steps_per_epoch,
                                validation_data=validation_dataset,
                                callbacks=[cp_callback, tensorboard_callback, lr_ctrl_callback, es_callback,
                                           MetricCollectorCallback()])
    return _model, _model_history
