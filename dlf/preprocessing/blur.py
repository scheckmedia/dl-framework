import tensorflow as tf
import tensorflow_probability as tfp

from dlf.core.registry import register_preprocessing_method
from dlf.core.preprocessing import PreprocessingMethod
from dlf.utils.helpers import random_apply


# https://stackoverflow.com/questions/52012657/how-to-make-a-2d-gaussian-filter-in-tensorflow
@register_preprocessing_method('blur', 'random_blur', 'Blur', 'RandomBlur')
class RandomBlur(PreprocessingMethod):
    """Data augmentation method that applies randomly a Gaussian blur to an image

    # Aliases
        - blur
        - random_blur
        - Blur
        - RandomBlur

    # Arguments
        size: float. Size of the convolutional kernel. Defaults to 3.
        mean: float. Mean value, centre of the distribution. Defaults to 0.
        std: float. Standard deviation or width of the distribution. Defaults to 5.

    # YAML Configuration

    ```yaml
        preprocess_list:
            blur:
                size: 5
                mean: 0.0
                std: 6
    ```
    """

    def __init__(self, size=3, mean=0, std=5):
        super().__init__()
        self.size = size // 2
        self.mean = mean
        self.std = std

    @tf.function
    def _build_kernel(self):
        # add a small epsilon for the rare case of delta = 0
        delta = tf.random.uniform([]) + tf.keras.backend.epsilon()

        d = tfp.distributions.Normal(
            self.mean, self.std * delta, validate_args=True)
        vals = d.prob(tf.range(start=-self.size,
                               limit=self.size + 1, dtype=tf.float32))
        gauss_kernel = tf.einsum('i,j->ij', vals, vals)
        gauss_kernel = (gauss_kernel / tf.reduce_sum(gauss_kernel)
                        )[:, :, tf.newaxis, tf.newaxis]
        gauss_kernel = tf.tile(gauss_kernel, [1, 1, 3, 1])
        return gauss_kernel

    @tf.function
    def _apply_gauss(self, image):
        kernel = self._build_kernel()
        image = tf.cast(image, tf.float32)
        image = tf.nn.depthwise_conv2d(tf.expand_dims(
            image, 0), kernel, strides=[1, 1, 1, 1], padding='SAME')
        image = tf.clip_by_value(
            image, clip_value_min=0.0, clip_value_max=255.0)
        return tf.squeeze(image, 0)

    def __call__(self, job):
        job.image = random_apply(self._apply_gauss, job.image, 0.5)
        return job
