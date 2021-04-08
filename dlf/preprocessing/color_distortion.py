import tensorflow as tf

from dlf.core.registry import register_preprocessing_method
from dlf.core.preprocessing import PreprocessingMethod
from dlf.utils.helpers import random_apply


@register_preprocessing_method('color_distortion', 'ColorDistortion', 'random_color_distortion', 'RandomColorDistortion')
class RandomColorDistortion(PreprocessingMethod):
    """Data augmentation method that randomly distorts colors of an RGB image

    # Aliases
        - color_distortion
        - ColorDistortion
        - random_color_distortion
        - RandomColorDistortion

    # Arguments
        s: float. Scalar that scales the distortion effect. Defaults to 1.0.
        color_jitter_probability: float [0,1]. Probability that color jittering is applied. Defaults to 0.8
        color_drop_probability: float [0,1]. Probability that color drop (RGB to Gray) is applied. Defaults to 0.8

    # YAML Configuration

    ```yaml
        preprocess_list:
            color_distortion:
                s: 0.7
                color_jitter_probability: 0.5
                color_drop_probability: 0.3
    ```

    # References
        - SimCLR: https://arxiv.org/abs/2002.05709
    """

    def __init__(self, s=1.0, color_jitter_probability=0.8, color_drop_probability=0.2):
        super().__init__()
        self.s = s
        self.color_jitter_probability = color_jitter_probability
        self.color_drop_probability = color_drop_probability

    @tf.function
    def color_jitter(self, x):
        x = tf.image.random_contrast(x, 1 - 0.8 * self.s, 1 + 0.8 * self.s)
        x = tf.image.random_saturation(x, 1 - 0.8 * self.s, 1 + 0.8 * self.s)
        x = tf.image.random_brightness(x, 0.8 * self.s)
        x = tf.image.random_hue(x, 0.2 * self.s)
        x = tf.clip_by_value(x, 0, 1)
        return x

    @tf.function
    def color_drop(self, x):
        x = tf.image.rgb_to_grayscale(x)
        x = tf.tile(x, [1, 1, 3])
        return x

    def __call__(self, job):
        job.image /= 255.0
        job.image = random_apply(self.color_jitter, job.image,
                                 self.color_jitter_probability)
        job.image = random_apply(self.color_drop, job.image,
                                 self.color_drop_probability)
        job.image *= 255.0
        return job
