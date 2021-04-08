import tensorflow as tf

from dlf.core.registry import register_preprocessing_method
from dlf.core.preprocessing import PreprocessingMethod


@register_preprocessing_method('resize', 'Resize')
class Resize(PreprocessingMethod):
    """Data augmentation method that resizes the dimension of a RGB image

    For details [tf.image.resize](https://www.tensorflow.org/api_docs/python/tf/image/resize)

    # Aliases
        - resize
        - Resize

    # Arguments
        width: int. Width of the resized image
        height: int. Height of the resized image
        method: ResizeMethod. Method used for resizing. Defaults to tf.image.ResizeMethod.BILINEAR.
        preserve_aspect_ratio: bool. If true, aspect ratio is preseverd. Defaults to False.
        antialias: bool. If true, an anti-aliasing filter is used for downsampling. Defaults to False.

    # YAML Configuration

        ```yaml
        preprocess_list:
            resize:
                width: 512
                height: 512
        ```
    """

    def __init__(self, width, height, method=tf.image.ResizeMethod.BILINEAR, preserve_aspect_ratio=False, antialias=False):
        super().__init__()
        self.output_shape = (height, width)
        self.method = method
        self.preserve_aspect_ratio = preserve_aspect_ratio
        self.antialias = antialias

    def __call__(self, job):
        job.image = tf.image.resize(job.image, self.output_shape, self.method,
                                    self.preserve_aspect_ratio, self.antialias)

        if type(job.mask) == tf.Tensor:
            job.mask = tf.image.resize(job.mask, self.output_shape, tf.image.ResizeMethod.NEAREST_NEIGHBOR,
                                       self.preserve_aspect_ratio, False)

        return job
