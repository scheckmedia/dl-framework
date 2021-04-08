import tensorflow as tf

from dlf.core.registry import register_preprocessing_method
from dlf.core.preprocessing import PreprocessingMethod
from dlf.utils.helpers import random_apply


@register_preprocessing_method('noise', 'Noise', 'RandomNoise', 'random_noise')
class RandomNoise(PreprocessingMethod):
    """Data augmentation method that randomly adds Gaussian noise to a RGB image

    # Aliases
        - noise
        - random_noise
        - Noise
        - RandomNoise

    # Arguments
        mean: float. Mean value, centre of the distribution. Defaults to 0.
        std: float. Standard deviation or width of the distribution. Defaults to 5.
        color: bool. If true, noise is added to color channels separately. Defaults to True.

    # YAML Configuration

    ```yaml
        preprocess_list:
            noise:
                mean: 0.0
                std: 0.8
                color: False
        ```
    """

    def __init__(self, mean=0.0, std=5.0, color=True):
        super().__init__()

        self.color = color
        self.mean = mean
        self.std = std

    def __call__(self, job):
        job.image = random_apply(self._apply_noise, job.image, 0.2)
        return job

    @tf.function
    def _apply_noise(self, image):
        image_shape = tf.shape(image)
        delta = tf.random.get_global_generator().uniform([1])

        if not self.color:
            image_shape = image_shape[:-1]

        noise = tf.random.normal(
            image_shape, self.mean, self.std * delta, dtype=tf.float32)

        if not self.color:
            noise = tf.stack([noise, noise, noise], axis=-1)

        image = tf.cast(image, tf.float32)
        image = tf.add(image, noise)
        image = tf.clip_by_value(
            image, clip_value_min=0.0, clip_value_max=255.0)
        return image
