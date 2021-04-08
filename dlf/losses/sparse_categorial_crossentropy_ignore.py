import tensorflow as tf
from dlf.core.registry import register_loss


@register_loss('SparseCategoricalCrossentropyIgnore', 'sparse_categorical_crossentropy_ignore')
class SparseCategoricalCrossentropyIgnore(tf.keras.losses.Loss):
    """Implementation for sparse crossentropy loss with ignore label functionality.

    The same implmentation like tf.keras.losses.SparseCategoricalCrossentropy but
    with the addtion that this implementation ignores all label ids <= num_classes.

    # Aliases
        - SparseCategoricalCrossentropyIgnore
        - sparse_categorical_crossentropy_ignore

    # Arguments
        num_classes: int. Number of classes, everthing above or equal will be ignored
        name: str, optonal, Name of the loss function. Defaults to 'sparse_categorical_crossentropy_ignore'.
        from_logits: bool.
            If true, y_pred is expected to be a logits tensor. Defaults to False.

    # Returns
        A tf.keras.losses.Loss

    # YAML Configuration
        ```yaml
        loss:
            SparseCategoricalCrossentropyIgnore:
                num_classes: 7
                from_logits: False
        ```
    """

    def __init__(self, num_classes, name='sparse_categorical_crossentropy_ignore',
                 from_logits=False):
        super().__init__(name=name)
        self.num_classes = num_classes
        self.from_logits = from_logits

    def call(self, y_true, y_pred):
        #y_pred = tf.argmax(y_pred, axis=-1)

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

        return tf.keras.losses.sparse_categorical_crossentropy(
            y_true, y_pred, from_logits=self.from_logits)
