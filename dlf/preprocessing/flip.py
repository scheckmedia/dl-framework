import tensorflow as tf

from dlf.core.registry import register_preprocessing_method
from dlf.core.preprocessing import PreprocessingMethod
from dlf.utils.helpers import random_apply

#https://github.com/tensorflow/models/blob/master/research/object_detection/core/preprocessor.py#L463
@register_preprocessing_method('h_flip', 'horizontal_flip', 'HFlip', 'HorizontalFlip')
class RandomHorizontalFlip(PreprocessingMethod):
    """Data augmentation method that randomly horizontal flips a RGB image

    # Aliases
        - h_flip
        - horizontal_flip
        - HFlip
        - HorizontalFlip

    # YAML Configuration

    ```yaml
        preprocess_list:
            h_flip:
    ```
    """

    def __init__(self):
        super().__init__()

    @tf.function
    def _apply_flip(self, image, mask, boxes):
        image = tf.image.flip_left_right(image)

        if mask is not None:
            mask = tf.image.flip_left_right(mask)

        if boxes is not None:
            ymin, xmin, ymax, xmax = tf.split(value=boxes, num_or_size_splits=4, axis=-1)
            flipped_xmin = tf.subtract(1.0, xmax)
            flipped_xmax = tf.subtract(1.0, xmin)
            boxes = tf.concat([ymin, flipped_xmin, ymax, flipped_xmax], axis=-1)

        return image, mask, boxes

    def __call__(self, job):
        job.image, job.mask, job.boxes = random_apply(self._apply_flip, (job.image, job.mask, job.boxes), 0.5)
        return job


@register_preprocessing_method('v_flip', 'vertical_flip', 'VFlip', 'VerticalFlip')
class RandomVerticalFlip(PreprocessingMethod):
    """Data augmentation method that randomly vertical flips a RGB image

    # Aliases
        - v_flip
        - vertical_flip
        - VFlip
        - VerticalFlip

    # YAML Configuration

    ```yaml
        preprocess_list:
            v_flip:
    ```
    """

    def __init__(self):
        super().__init__()

    @tf.function
    def _apply_flip(self, image, mask, boxes):
        image = tf.image.flip_up_down(image)

        if mask is not None:
            mask = tf.image.flip_up_down(mask)

        if boxes is not None:
            ymin, xmin, ymax, xmax = tf.split(value=boxes, num_or_size_splits=4, axis=1)
            flipped_ymin = tf.subtract(1.0, ymax)
            flipped_ymax = tf.subtract(1.0, ymin)
            boxes = tf.concat([flipped_ymin, xmin, flipped_ymax, xmax], 1)

        return image, mask, boxes

    def __call__(self, job):
        job.image, job.mask, job.boxes = random_apply(self._apply_flip, (job.image, job.mask, job.boxes), 0.5)
        return job
