import tensorflow as tf

from dlf.core.registry import register_preprocessing_method
from dlf.core.preprocessing import PreprocessingMethod
from dlf.utils.helpers import random_apply


@register_preprocessing_method('brightness', 'Brightness', 'random_brightness', 'RandomBrightness')
class RandomBrightness(PreprocessingMethod):
    """Data augmentation method that randomly adjust the brightness of an RGB image

    # Aliases
        - brightness
        - Brightness
        - random_brightness
        - RandomBrightness

    # Arguments
        max_delta: float [0,1). Scalar that is added to the pixel. Defaults to 0.2.

    # YAML Configuration

    ```yaml
        preprocess_list:
            brigthness:
                max_delta: 0.2
    ```

    """

    def __init__(self, max_delta=0.2):
        super().__init__()
        self.delta = tf.random.uniform([], maxval=max_delta)

    @tf.function
    def _apply_brightnes(self, image):
        image = tf.image.adjust_brightness(image / 255, self.delta) * 255
        image = tf.clip_by_value(
            image, clip_value_min=0.0, clip_value_max=255.0)
        return image

    def __call__(self, job):
        job.image = random_apply(self._apply_brightnes, job.image, 0.5)
        return job
