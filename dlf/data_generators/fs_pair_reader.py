from glob import glob
import tensorflow as tf
import numpy as np

from dlf.core.data_generator import DataGenerator
from dlf.core.registry import register_data_generator
from dlf.core.builder import build_preprocessing_exectutor


@register_data_generator('FsRandomUnpairedReader', 'fs_random_unpaired_reader')
class FsRandomUnpairedReader(DataGenerator):
    """A data generator which generates random pairs of images

    # Aliases
        - FsRandomUnpairedReader
        - fs_random_unpaired_reader

    # Arguments
        paths_lhs: list of str. List of glob patterns to find images for the left hand side
        paths_rhs: list of str. List of glob patterns to find images for the right hand side
        channels: int. Number of channels which each image should contain. Defaults to 3.
        lhs_limit: int. If specified N elements of lhs set are selected. Defaults to None.
        rhs_limit: int. If specified N elements of rhs set are selected. Defaults to None.
        preprocess_list: dict. Dictionary where the value describes a preprocessing method and the values are the arguments. Defaults to None.
        shuffle_buffer: int.
            Size of shuffle buffer. For details check: [tf.data.Dataset](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#shuffle). Defaults to 2000.

    # Raises
        FileNotFoundError: If one path of the lhs path set is not valid
        FileNotFoundError: If one path of the rhs path set is not valid
        ValueError: If crop or resize is not specified in preprocessing pipeline

    # YAML Configuration
    ```yaml
    input_reader:
        training_reader:
            name: fs_random_unpaired_reader
            paths_lhs: /mnt/data/datasets/theodore_wheeled_walker/*_img.png
            paths_rhs:
                - /mnt/data/datasets/omnidetector-Flat/JPEGImages/*.jpg
                - /mnt/data/datasets/indoor_cvpr/Images/**/*.jpg
            lhs_limit: 20000
            rhs_limit: 20000
            shuffle_buffer: 400
            preprocess_list:
                crop:
                    width: 512
                    height: 512
    ```
    """

    def __init__(self, paths_lhs, paths_rhs, channels=3, lhs_limit=None, rhs_limit=None, preprocess_list=None, shuffle_buffer=100):
        if isinstance(paths_lhs, str):
            paths_lhs = [paths_lhs]

        if isinstance(paths_rhs, str):
            paths_rhs = [paths_rhs]

        files_lhs = self._parse_paths(paths_lhs, 'Lhs')
        files_rhs = self._parse_paths(paths_rhs, 'Rhs')

        tf.get_logger().info("Found {} images for LHS".format(len(files_lhs)))
        tf.get_logger().info("Found {} images for RHS".format(len(files_rhs)))

        np.random.shuffle(files_lhs)
        np.random.shuffle(files_rhs)

        if lhs_limit is not None:
            files_lhs[:lhs_limit]

        if rhs_limit is not None:
            files_rhs[:rhs_limit]

        if len(files_lhs) == 0:
            raise FileNotFoundError(
                'No files for paths_lsh pattern "{}" where found'.format(paths_lhs))

        if len(files_rhs) == 0:
            raise FileNotFoundError(
                'No files for paths_rhs pattern "{}" where found'.format(paths_rhs))

        if preprocess_list is None or ('resize' not in preprocess_list and 'crop' not in preprocess_list):
            raise ValueError(
                'FsRandomPairReader requires images of the same size. Add the "resize" operation to preprocessing list')

        preprocessing_exectutor = build_preprocessing_exectutor(
            preprocess_list)
        indices_lhs = np.arange(0, len(files_lhs))
        indices_rhs = np.arange(0, len(files_rhs))

        def generator():
            while True:
                lhs = np.random.choice(indices_lhs, 1)[0]
                rhs = np.random.choice(indices_rhs, 1)[0]

                bytes_lhs = tf.io.read_file(files_lhs[lhs])
                bytes_rhs = tf.io.read_file(files_rhs[rhs])

                image_lhs = tf.io.decode_image(
                    bytes_lhs, channels=channels, dtype=tf.uint8, expand_animations=False)
                image_rhs = tf.io.decode_image(
                    bytes_rhs, channels=channels, dtype=tf.uint8, expand_animations=False)

                # image_lhs = tf.squeeze(image_lhs)
                # image_rhs = tf.squeeze(image_rhs)

                job_lhs = preprocessing_exectutor(image_lhs)
                job_rhs = preprocessing_exectutor(image_rhs)

                yield job_lhs.image, job_rhs.image

        ds = tf.data.Dataset.from_generator(
            generator, output_types=(tf.float32, tf.float32))
        ds = ds.map(lambda x, y: {'x_batch': x, 'y_batch': y},
                    num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds = ds.shuffle(shuffle_buffer)

        super().__init__(self.__class__.__name__, ds, None)

    def _parse_paths(self, paths, alias):
        files = []
        for path in paths:
            current_files = glob(path)
            if len(current_files) == 0:
                raise FileNotFoundError(
                    "{} path {} not found!".format(alias, path))

            files += current_files

        return files
