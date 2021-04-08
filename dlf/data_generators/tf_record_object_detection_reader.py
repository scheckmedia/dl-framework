from pathlib import Path
import tensorflow as tf
from glob import glob

from dlf.core.data_generator import DataGenerator
from dlf.utils.helpers import parse_label_map
from dlf.core.registry import register_data_generator
from dlf.core.builder import build_preprocessing_exectutor


@register_data_generator('TfRecordSSDReader', 'tf_record_ssd_reader')
class TfRecordSSDReader(DataGenerator):
    """A data generator which uses the TFRecord format for object detection tasks using SSD architectures

    # Arguments
        path: str. Path to TFRecord file. Defaults to None.
        glob_pattern: str. Glob pattern for multiple TFRecord files. This is the case if a record file was sharded over multiple files. Defaults to None.
        labelmap: str. Path to labelmap related to TFRecord file
        image_key: str. Key where the images are located in tf record example. Defaults to 'image/encoded'.
        bounding_box_key: str. Key where the bounding boxes are located in tf record example Defaults to 'image/object/bbox/'.
        label_key: str. Key where the labes are located. Defaults to 'image/object/class/label'.
        id_key: str. Key where the images ids are lcoated. Defaults to 'image/source_id'.
        background_as_zero: bool. If true, background with 0 zero will be added to label list. Defaults to True.
        shuffle: bool. If true, dataset is shuffled. Defaults to True.
        ignore: list of int. A list of integers where each number is ignored during validation e.g. label 255 in PascalVOC. Defaults to None.
        remap: dict[int, int]. Dictionary where the key is the source category which is remaped to the value, target id. Defaults to None.
        preprocess_list: dict. Dictionary where the value describes a preprocessing method and the values are the arguments. Defaults to None.
        shuffle_buffer: int. Size of shuffle buffer. For details check: [tf.data.Dataset](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#shuffle). Defaults to 2000.

    # Raises
        FileNotFoundError: If TFRecord file is not found
        FileNotFoundError: If Labelmap file is not found

    # YAML Configuration
        Sample configuration for a object detection reader that applies remapping and uses a list of pre-processing functions before passing the image to CNN.

        ```yaml
        input_reader:
            training_reader:
                name: tf_record_ssd_reader
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

    def __init__(self, labelmap, path=None, glob_pattern=None,
                 image_key='image/encoded',
                 bounding_box_key='image/object/bbox/',
                 label_key='image/object/class/label',
                 id_key='image/source_id',
                 background_as_zero=True,
                 shuffle=True, ignore=None, remap=None, preprocess_list=None, shuffle_buffer=2000):

        if path is not None and not Path(path).exists():
            raise FileNotFoundError("Dataset {} not found!".format(path))

        if glob is not None and len(glob(glob_pattern)) == 0:
            raise FileNotFoundError(
                "Datasets with pattern {} not found!".format(glob_pattern))
        else:
            path = tf.data.Dataset.list_files(glob_pattern)

        if not Path(labelmap).exists():
            raise FileNotFoundError("Label map {} not found!".format(labelmap))

        self.__keys = {
            'img': image_key,
            'bbox': bounding_box_key,
            'label': label_key,
            'id': id_key
        }

        for k in ['xmin', 'xmax', 'ymin', 'ymax']:
            self.__keys["%s_bbox" % k] = bounding_box_key + k

        labels = parse_label_map(labelmap)

        if background_as_zero:
            s = {0: "background"}
            labels = {**s, **labels}

        self.label_index_map = {}
        transformed_labels = {}
        keys = list(labels.keys())
        for k, v in labels.items():
            self.label_index_map[k] = keys.index(k)
            transformed_labels[keys.index(k)] = v

        labels = transformed_labels

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

        padded_out_shape = {
            'x_batch': [None, None, None],
            'y_batch': {
                'ids': [None, ],
                'gt_boxes': [None, 4],
                'gt_labels': [None, ]
            }
        }

        padded_out_values = {
            'x_batch': None,
            'y_batch': {
                'ids': None,
                'gt_boxes': -1.0,
                'gt_labels': -1.0
            }
        }

        super().__init__(self.__class__.__name__,
                         dataset, labels, padded_out_shape, padded_out_values)

    def _parse_feature(self, string_record, ignore, remap, preprocessing_exectutor):
        feature = {
            'image/height': tf.io.FixedLenFeature([], tf.int64),
            'image/width': tf.io.FixedLenFeature([], tf.int64),
            self.__keys['img']: tf.io.FixedLenFeature([], tf.string),
            self.__keys['id']: tf.io.FixedLenFeature([], tf.string),
            self.__keys['xmin_bbox']: tf.io.VarLenFeature(tf.float32),
            self.__keys['xmax_bbox']: tf.io.VarLenFeature(tf.float32),
            self.__keys['ymin_bbox']: tf.io.VarLenFeature(tf.float32),
            self.__keys['ymax_bbox']: tf.io.VarLenFeature(tf.float32),
            self.__keys['label']: tf.io.VarLenFeature(tf.int64)
        }

        example = tf.io.parse_single_example(string_record, feature)

        image = tf.io.decode_image(example[self.__keys['img']], dtype=tf.uint8, channels=3, expand_animations=False)
        image = tf.cast(image, tf.float32)

        ids = example[self.__keys['id']]
        ids = tf.reshape(ids, (-1,))

        xmin = tf.sparse.to_dense(example[self.__keys['xmin_bbox']])
        ymin = tf.sparse.to_dense(example[self.__keys['ymin_bbox']])
        xmax = tf.sparse.to_dense(example[self.__keys['xmax_bbox']])
        ymax = tf.sparse.to_dense(example[self.__keys['ymax_bbox']])

        boxes = tf.stack([ymin, xmin, ymax, xmax], axis=1)

        labels = tf.sparse.to_dense(example[self.__keys['label']])

        new_labels = tf.zeros_like(labels, dtype=tf.int32)
        for k, v in self.label_index_map.items():
            change = tf.where(
                tf.equal(labels, k),
                v, 0
            )
            new_labels = tf.add(new_labels, change)

        labels = tf.cast(new_labels, tf.float32)

        # if remap is not None:
        #     for k, v in remap.items():
        #         k = tf.cast(k, tf.uint8)
        #         v = tf.cast(v, tf.uint8)
        #         mask = tf.where(tf.equal(mask, k), v, mask)

        job = preprocessing_exectutor(image, None, boxes)

        out = {
            'x_batch': job.image,
            'y_batch': {
                'ids': ids,
                'gt_boxes': job.boxes,
                'gt_labels': labels
            }
        }
        return out
