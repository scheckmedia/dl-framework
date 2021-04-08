import tensorflow as tf

from dlf.core.registry import register_preprocessing_method
from dlf.core.preprocessing import PreprocessingMethod
from dlf.utils.helpers import random_apply


@register_preprocessing_method('saturation', 'Saturation', 'random_saturation', 'RandomSaturation')
class RandomSaturation(PreprocessingMethod):
    """Data augmentation method that randomly adjust the saturation of a RGB image

    # Aliases
        - saturation
        - Saturation
        - RandomStaturation
        - random_staturation

    # Arguments
        lower: float. Lowest value that is added to the saturation channel of a HSV converted image
        upper: float. Highest value that is added to the saturation channel of a HSV converted image

    # YAML Configuration

        ```yaml
        preprocess_list:
            saturation:
                lower: 0.3
                upper: 1.5
        ```
    """

    def __init__(self, lower=0.8, upper=1.25):
        super().__init__()
        self.lower = lower
        self.upper = upper

    @tf.function
    def _apply_saturation(self, image):
        factor = tf.random.uniform([], self.lower, self.upper)
        image = tf.image.adjust_saturation(image / 255, factor) * 255
        image = tf.clip_by_value(
            image, clip_value_min=0.0, clip_value_max=255.0)
        return image

    def __call__(self, job):
        job.image = random_apply(self._apply_saturation, job.image, 0.5)
        return job
