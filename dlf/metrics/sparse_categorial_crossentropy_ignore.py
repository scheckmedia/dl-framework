import tensorflow as tf
from dlf.core.registry import register_metric


@register_metric('SparseCategoricalCrossentropyIgnore', 'sparse_categorical_crossentropy_ignore')
class SparseCategoricalCrossentropyIgnore(tf.keras.metrics.SparseCategoricalCrossentropy):
    """Implementation for sparse crossentropy metric with ignore label functionality.

    The same implmentation like tf.keras.metrics.SparseCategoricalCrossentropy but
    with the addtion that this implementation ignores all label ids <= num_classes.

    # Arguments
        num_classes: int. Number of classes, everthing above will be ignored
        name: str, optonal, Name of the loss function. Defaults to 'sparse_categorical_crossentropy_ignore'.
        from_logits: bool.
            If true, y_pred is expected to be a logits tensor. Defaults to False.

    # Returns
        A tf.keras.metrics.Metric

    # YAML Configuration
        ```yaml
        metrics:
            SparseCategoricalCrossentropyIgnore:
                from_logits: False
                num_classes: 7
        ```
    """

    def __init__(self, num_classes, from_logits=False, name='sparse_categorical_crossentropy_ignore', **kwargs):
        super().__init__(name=name, from_logits=from_logits, **kwargs)
        self.num_classes = num_classes

    def update_state(self, y_true, y_pred, sample_weight=None):
        classes = y_pred.shape[-1]

        if y_pred.shape.ndims > 1:
            y_pred = tf.reshape(y_pred, [-1, classes])

        if y_true.shape.ndims > 1:
            y_true = tf.reshape(y_true, [-1])

        # ignore all labels >= num_classes
        valid_indices = tf.where(y_true < self.num_classes)
        y_true = tf.squeeze(tf.gather(y_true, valid_indices))
        y_pred = tf.squeeze(tf.gather(y_pred, valid_indices))

        y_true = tf.cast(y_true, tf.int64)
        super().update_state(y_true, y_pred, sample_weight)
