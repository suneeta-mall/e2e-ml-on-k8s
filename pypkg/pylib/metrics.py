import logging
import tensorflow as tf


def accuracy(truth, pred):
    m = tf.keras.metrics.BinaryAccuracy()
    m.update_state(truth, pred)
    return m.result().numpy()


def mean_iou(truth, pred):
    m = tf.keras.metrics.MeanIoU(3)
    m.update_state(truth, pred)
    return m.result().numpy()


def precision(truth, pred):
    m = tf.keras.metrics.Precision()
    m.update_state(truth, pred)
    return m.result().numpy()


def iou_score(truth, pred, weights=1., label_smoothing=0, reduction=tf.keras.losses.Reduction.AUTO):
    # IoU/Jaccard score in range [0, 1]
    # reduction if true is applied per image otherwise summed over batches.
    axes = [1, 2] if reduction else [0, 1, 2]

    truth = tf.cast(truth, tf.float32)

    intersection = tf.keras.backend.sum(truth * pred, axis=axes)
    union = tf.keras.backend.sum(truth + pred, axis=axes) - intersection
    iou = (intersection + label_smoothing) / (union + label_smoothing)

    # mean per image
    if reduction:
        iou = tf.keras.backend.mean(iou, axis=0)
    # weighted mean per class
    iou = tf.keras.backend.mean(iou * weights)
    return iou


def bce_jaccard_loss(truth, pred, bce_weight=1., weights=1., label_smoothing=0,
                     reduction=tf.keras.losses.Reduction.AUTO):
    truth = tf.cast(truth, tf.float32)
    # bce_loss = tf.keras.losses.BinaryCrossentropy(label_smoothing=label_smoothing, reduction=reduction)
    # bce = bce_loss.call(truth, pred) * weights
    bce = tf.keras.losses.binary_crossentropy(truth, pred, label_smoothing=label_smoothing) * weights
    bce = tf.keras.backend.mean(bce)
    jaccard_score = iou_score(truth, pred, label_smoothing=label_smoothing, weights=weights, reduction=reduction)
    jaccard_loss = 1 - jaccard_score
    return bce_weight * bce + jaccard_loss


def f_score(truth, pred, weights=1, beta=1, label_smoothing=0, reduction=tf.keras.losses.Reduction.AUTO):
    axes = [1, 2] if reduction else [0, 1, 2]

    truth = tf.cast(truth, tf.float32)

    tp = tf.keras.backend.sum(truth * pred, axis=axes)
    fp = tf.keras.backend.sum(pred, axis=axes) - tp
    fn = tf.keras.backend.sum(truth, axis=axes) - tp

    score = ((1 + beta ** 2) * tp + label_smoothing) \
            / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + label_smoothing)

    # mean per image
    if reduction:
        score = tf.keras.backend.mean(score, axis=0)

    # weighted mean per class
    score = tf.keras.backend.mean(score * weights)
    return score


def f1_score(truth, pred, weights=1, label_smoothing=0, reduction=tf.keras.losses.Reduction.AUTO):
    return f_score(truth, pred, weights=weights, beta=1, label_smoothing=label_smoothing, reduction=reduction)


class BCEJaccardLoss():
    def __init__(self, from_logits=False, label_smoothing=0, reduction=tf.keras.losses.Reduction.AUTO,
                 name="bce_jaccard_loss"):
        self.reduction = reduction
        self.name = name
        self.label_smoothing = label_smoothing
        self.from_logits = from_logits

    def __call__(self, y_true, y_pred, sample_weight=None):
        return self.call(y_true, y_pred)

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def get_config(self):
        return {'reduction': self.reduction, 'name': self.name}

    def call(self, y_true, y_pred):
        return bce_jaccard_loss(y_pred, y_true,
                                label_smoothing=self.label_smoothing, reduction=self.reduction)

    def _get_reduction(self):
        return self.reduction


class MetricCollectorCallback(tf.keras.callbacks.Callback):
    # https://www.kubeflow.org/docs/components/hyperparameter-tuning/hyperparameter/#metrics-collector
    def on_epoch_end(self, epoch, logs=None):
        # print(f"epoch {epoch + 1}:\nloss={logs['loss']}\niou_score={logs['iou_score']}\n"
        #       f"f1_score={logs['f1_score']}\nbinary_accuracy={logs['binary_accuracy']}\n"
        #       f"val_loss={logs['val_loss']}\nval_iou_score={logs['val_iou_score']}\n"
        #       f"val_f1_score={logs['val_f1_score']}\nval_binary_accuracy={logs['val_binary_accuracy']}\n")

        # INFO:root:Epoch[18] Train-accuracy=0.997889
        # INFO:root:Epoch[18] Time cost=7.423
        # INFO:root:Epoch[18] Validation-accuracy=0.979797
        logging.info(f"Epoch[{epoch + 1}] loss={logs['loss']})")
        logging.info(f"Epoch[{epoch + 1}] iou_score={logs['iou_score']}")
        logging.info(f"Epoch[{epoch + 1}] f1_score={logs['f1_score']})")
        logging.info(f"Epoch[{epoch + 1}] binary_accuracy={logs['binary_accuracy']}")
        logging.info(f"Epoch[{epoch + 1}] val_loss={logs['val_loss']}")
        logging.info(f"Epoch[{epoch + 1}] val_iou_score={logs['val_iou_score']}")
        logging.info(f"Epoch[{epoch + 1}] val_f1_score={logs['val_f1_score']}")
        logging.info(f"Epoch[{epoch + 1}] val_binary_accuracy={logs['val_binary_accuracy']}")
