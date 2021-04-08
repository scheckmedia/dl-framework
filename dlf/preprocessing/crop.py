import tensorflow as tf

from dlf.core.registry import register_preprocessing_method
from dlf.core.preprocessing import PreprocessingMethod


@register_preprocessing_method('crop', 'Crop', 'RandomCrop', 'random_crop')
class RandomCrop(PreprocessingMethod):
    """Data augmentation method that randomly crops an RGB image

    # Aliases
        - crop
        - Crop
        - RandomCrop
        - random_crop

    # Arguments
        width: int. Width of the cropped image
        height: int. Height of the cropped image

    # YAML Configuration

    ```yaml
        preprocess_list:
            crop:
                width: 512
                height: 512
    ```
    """

    def __init__(self, width, height, scale=1.5):

        super().__init__()
        self.width = width
        self.height = height
        self.scale = scale

    # https://stackoverflow.com/questions/42147427/tensorflow-how-to-randomly-crop-input-images-and-labels-in-the-same-way
    def __call__(self, job):

        if type(job.mask) == tf.Tensor:
            job.image, job.mask = self._crop_synchron(job.image, job.mask)
        else:
            job.image = self._crop(job.image, self.height, self.width)

        return job

    @tf.function
    def resize_image_keep_aspect(self, image, lo_dim=224):
        # Take width/height
        initial_width = tf.shape(image)[2]
        initial_height = tf.shape(image)[1]

        # Take the greater value, and use it for the ratio
        min_ = tf.minimum(initial_width, initial_height)
        ratio = tf.cast(min_, tf.float32) / lo_dim

        new_width = tf.cast(
            tf.cast(initial_width, tf.float32) / ratio, tf.int32)
        new_height = tf.cast(
            tf.cast(initial_height, tf.float32) / ratio, tf.int32)

        return tf.image.resize(image, [new_height, new_width])

    @tf.function
    def _crop(self, image, height, width):
        shape = tf.TensorShape([height, width])
        if image.get_shape()[:2] != shape:
            low_dim = tf.reduce_min(
                [height * self.scale, width * self.scale])
            image = self.resize_image_keep_aspect(image, low_dim)

        if image.get_shape().ndims == 3:
            shape = [height, width, 3]

        image = tf.image.random_crop(image, shape)
        return image

    @tf.function
    def _crop_synchron(self, image, mask):
        combined = tf.concat([image, mask], axis=2)
        image_shape = tf.shape(image)
        combined_pad = tf.image.pad_to_bounding_box(
            combined, 0, 0,
            tf.maximum(self.height, image_shape[0]),
            tf.maximum(self.width, image_shape[1]))
        last_label_dim = tf.shape(mask)[-1]
        last_image_dim = tf.shape(image)[-1]
        combined_crop = tf.image.random_crop(
            combined_pad,
            size=tf.concat([(self.height, self.width), [last_label_dim + last_image_dim]],
                           axis=0))
        image = combined_crop[:, :, :last_image_dim]
        mask = combined_crop[:, :, last_image_dim:]
        return image, mask
