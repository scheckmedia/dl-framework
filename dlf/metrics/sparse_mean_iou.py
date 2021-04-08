import tensorflow as tf
from dlf.core.registry import register_metric


@register_metric('sparse_mean_iou', 'SparseMeanIoU')
class SparseMeanIoU(tf.keras.metrics.MeanIoU):
    """Implementation of MeanIoU metric for sparse tensors

    This implementation allows it to use sparse tensors as y_pred for the calculation of the mIoU.
    In addtion, all labels <= num_classes will be ignored.

    # Arguments
        num_classes: int. Number of classes, everthing above or equal will be ignored
        name: str. Name of the metric. Defaults to 'sparse_mean_iou'.

    # YAML Configuration
        ```yaml
        metrics:
            SparseMeanIoU:
                num_classes: 7
        ```
    """

    def __init__(self, num_classes, name='sparse_mean_iou', **kwargs):
        super().__init__(name=name, num_classes=num_classes, **kwargs)
        self.num_classes = num_classes

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred, axis=-1)

        if y_pred.shape.ndims > 1:
            y_pred = tf.reshape(y_pred, [-1])

        if y_true.shape.ndims > 1:
            y_true = tf.reshape(y_true, [-1])

        # ignore all labels >= num_classes
        valid_indices = tf.where(y_true < self.num_classes)
        y_true = tf.gather(y_true, valid_indices)
        y_pred = tf.gather(y_pred, valid_indices)

        super().update_state(y_true, y_pred, sample_weight)

    def get_confusion_matrix(self):
        return self.total_cm
