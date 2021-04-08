from pathlib import Path
import tensorflow as tf
from PIL import Image
from io import BytesIO
import numpy as np

from dlf.core.data_generator import DataGenerator
from dlf.core.registry import register_data_generator
from dlf.core.builder import build_preprocessing_exectutor
from dlf.utils.helpers import parse_label_map


@register_data_generator('TfRecordSegmentationReader', 'tf_record_segmentation_reader')
class TfRecordSegmentationReader(DataGenerator):
    """A data generator for segmentation tasks which uses the TFRecord format

    # Arguments
        path: str. Path to TFRecord file
        labelmap: str. Path to labelmap related to TFRecord file
        background_as_zero: bool. If true, the ID 0 is discribing the background. Defaults to True.
        shuffle: bool. If true, dataset is shuffled. Defaults to True.
        ignore: list of int. A list of integers where each number is ignored during validation e.g. label 255 in PascalVOC. Defaults to None.
        remap: dict[int, int]. Dictionary where the key is the source category which is remaped to the value, target id. Defaults to None.
        preprocess_list: dict. Dictionary where the value describes a preprocessing method and the values are the arguments. Defaults to None.
        shuffle_buffer: int.
            Size of shuffle buffer. For details check: [tf.data.Dataset](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#shuffle). Defaults to 2000.

    # Raises
        FileNotFoundError: If TFRecord file is not found
        FileNotFoundError: If Labelmap file is not found

    # YAML Configuration
        Sample configuration for a segmentation reader that applies remapping and uses a list of pre-processing functions before passing the image to CNN.

        ```yaml
        input_reader:
            training_reader:
                name: tf_record_segmentation_reader
                path: /mnt/data/datasets/wheeled_walker_100k_8fps_25switch/25k_sample_4_mask/training.tfrecord
                labelmap: &labelmap /mnt/data/datasets/wheeled_walker_100k_8fps_25switch/label_map.pbtxt
                ignore:
                remap:
                    1:0
                    2:0
                    3:0
                    5:1
                    6:1
                    7:1
                preprocess_list:
                    h_flip:
                    v_flip:
                    resize:
                        output_shape:
                            - 512
                            - 512
                    saturation:
                    brightness:
                        max_delta: 0.6
                    blur:
                    noise:
                        mean: 0
                        std: 5
        ```
    """

    def __init__(self, path, labelmap, background_as_zero=True, shuffle=True, ignore=None, remap=None, preprocess_list=None, shuffle_buffer=2000):
        if not Path(path).exists():
            raise FileNotFoundError("Dataset {} not found!".format(path))

        if not Path(labelmap).exists():
            raise FileNotFoundError("Label map {} not found!".format(labelmap))

        labels = parse_label_map(labelmap)
        if background_as_zero:
            labels[0] = "background"

        if ignore and isinstance(ignore, (list,)):
            if len(ignore):
                labels[255] = 'void'

            for i in ignore:
                if i in labels:
                    labels.pop(i)

        if remap and isinstance(remap, (dict,)):
            for k, v in remap.items():
                labels.pop(k)

        preprocessing_exectutor = build_preprocessing_exectutor(
            preprocess_list)

        def mapping_wrapper(string_record):
            return self._parse_feature(string_record, ignore, remap, preprocessing_exectutor)

        dataset = tf.data.TFRecordDataset(path)
        dataset = dataset.map(
            mapping_wrapper, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        if shuffle:
            dataset = dataset.shuffle(shuffle_buffer)

        super().__init__(self.__class__.__name__, dataset, labels)

    def _parse_feature(self, string_record, ignore, remap, preprocessing_exectutor):
        feature = {
            'image/height': tf.io.FixedLenFeature([], tf.int64),
            'image/width': tf.io.FixedLenFeature([], tf.int64),
            'image/source_id': tf.io.FixedLenFeature([], tf.string),
            'image/encoded': tf.io.FixedLenFeature([], tf.string),
            'image/encoded_categories': tf.io.FixedLenFeature([], tf.string)
        }

        example = tf.io.parse_single_example(string_record, feature)

        def decode_indexed_png(example_bytes):
            mask = np.array(Image.open(
                BytesIO(example_bytes.numpy())), dtype=np.uint8)
            mask = np.expand_dims(mask, -1)
            return mask

        image = tf.io.decode_png(example['image/encoded'], 3, tf.uint8)
        # not working, tf can 't handle indexed pngs
        # mask = tf.io.decode_png(example['image/encoded_categories'], channels=0)

        # but py_function with numpy components cant use autograph, we need a faster way here
        mask = tf.py_function(decode_indexed_png, [
            example['image/encoded_categories']], Tout=tf.uint8)
        mask.set_shape(image.shape[:-1] + (1))

        image = tf.cast(image, tf.float32)

        if ignore is not None:
            for ignore in ignore:
                mask = tf.where(
                    tf.equal(mask, tf.cast(ignore, tf.uint8)),
                    255, mask
                )

        if remap is not None:
            for k, v in remap.items():
                k = tf.cast(k, tf.uint8)
                v = tf.cast(v, tf.uint8)
                mask = tf.where(tf.equal(mask, k), v, mask)

        mask = tf.cast(mask, tf.float32)

        job = preprocessing_exectutor(image, mask)
        mask = tf.squeeze(job.mask)
        return {'x_batch': job.image, 'y_batch': mask}
