import os
from glob import glob
import tensorflow as tf
import numpy as np

from dlf.core.data_generator import DataGenerator
from dlf.core.registry import register_data_generator
from dlf.core.builder import build_preprocessing_exectutor


@register_data_generator('FsImageNetLikeReader', 'fs_imagenet_like_reader')
class FsImageNetLikeReader(DataGenerator):
    """A data generator which reads imagenet like file structures

    ```
    - folder
        - folder class1
            - file 1
            - file 2
            - ...
        - folder class2
            - file 1
            - file 2
            - ...
        - ....
    ```

    # Aliases
        - FsImageNetLikeReader
        - fs_imagenet_like_reader

    # Arguments
        path: str. glob pattern to find images
        is_simclr: bool. If true, for each image a replicat with additional augmentation is generated. Defaults to False.
        categorical_labels: bool. If true, labes are one-hot encoded. Defaults to False.
        preprocess_list: dict. Dictionary where the value describes a preprocessing method and the values are the arguments. Defaults to None.
        shuffle_buffer: int.
            Size of shuffle buffer. For details check: [tf.data.Dataset](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#shuffle). Defaults to 2000.

    # Raises
        FileNotFoundError: If path is not valid

    # YAML Configuration
    ```yaml
    input_reader:
        training_reader:
            name: fs_imagenet_like_reader
            path: /mnt/data/datasets/openimages/semisupervised/filtered_62/train/**/*.jpg
            categorical_labels: True
            is_simclr: True
            preprocess_list:
            h_flip:
            color_distortion:
                s: 1.0
                color_jitter_probability: 0.5
                color_drop_probability: 0.3
            resize:
                width: 224
                height: 224
    ```
    """

    def __init__(self, path, is_simclr=False, categorical_labels=False, preprocess_list=None, shuffle=True, shuffle_buffer=100):
        self.categorical_labels = categorical_labels
        self.is_simclr = is_simclr

        file_list = np.array(glob(path))
        if len(file_list) == 0:
            raise FileNotFoundError(
                'No files for path pattern "{}" where found'.format(path))

        self.preprocessing_exectutor = build_preprocessing_exectutor(
            preprocess_list)

        indices = np.arange(0, len(file_list))
        classes = list(set([x.split(os.path.sep)[-2] for x in file_list]))
        self.num_classes = len(classes)

        label_mapping = {}
        for i, cls in enumerate(classes):
            label_mapping[cls] = i

        labels = np.array([label_mapping[x.split(os.path.sep)[-2]]
                           for x in file_list])

        ds = tf.data.Dataset.from_tensor_slices((file_list, labels)).map(
            lambda x, y: self._process_file_path(
                x, y),
            num_parallel_calls=tf.data.experimental.AUTOTUNE)

        if shuffle:
            ds = ds.shuffle(shuffle_buffer)

        super().__init__(self.__class__.__name__, ds, label_mapping)

    def _process_file_path(self, x, y):
        img = tf.io.read_file(x)
        img = tf.io.decode_image(img, channels=3, expand_animations=False)
        # if dtype float of decode_image then images are normlized!!!
        img = tf.cast(img, tf.float32)

        if self.categorical_labels:
            y = tf.one_hot(y, self.num_classes)

        job = self.preprocessing_exectutor(img)
        if self.is_simclr:
            second_job = self.preprocessing_exectutor(img)
            return {'x_batch': {
                'x1': job.image,
                'x2': second_job.image
            }, 'y_batch': y}

        return {'x_batch': job.image, 'y_batch': y}
