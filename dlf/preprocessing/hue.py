import tensorflow as tf

from dlf.core.registry import register_preprocessing_method
from dlf.core.preprocessing import PreprocessingMethod
from dlf.utils.helpers import random_apply


@register_preprocessing_method('hue', 'Hue', 'random_hue', 'RandomHue')
class RandomHue(PreprocessingMethod):
    """Data augmentation method that randomly adjust the hue of a RGB image

    # Aliases
        - hue
        - Hue
        - random_hue
        - RandomHue

    # Arguments
        max_delta: float [-1,1]. Scalar that is added to the hue. Defaults to 0.2.

    # YAML Configuration

    ```yaml
        preprocess_list:
            hue:
                max_delta: 0.2
        ```

    """

    def __init__(self, max_delta=0.2):
        super().__init__()
        self.delta = max_delta

    @tf.function
    def _apply_hue(self, image):
        image = tf.image.adjust_hue(image / 255, self.delta) * 255
        image = tf.clip_by_value(
            image, clip_value_min=0.0, clip_value_max=255.0)
        return image

    def __call__(self, job):
        job.image = random_apply(self._apply_hue, job.image, 0.5)
        return job
