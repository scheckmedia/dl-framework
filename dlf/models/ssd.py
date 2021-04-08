import tensorflow as tf

from dlf.core.registry import register_model
from dlf.core.model import ModelWrapper
from dlf.utils.box_utils import compute_target, generate_default_boxes, encode, decode, compute_nms
from dlf.models.ssd_backbone.vgg import VGG16_SSD
import numpy as np


@register_model('ssd', 'SSD')
class SSD(ModelWrapper):
    """A implementation for an Single Shot Detector (SSD)

    # Aliases
        - ssd
        - SSD

    # Architectures implemented
        - vgg_ssd_300: Original Paper implmentation of SSD 300 with VGG as backend
        - vgg_ssd_512: Original Paper implmentation of SSD 512 with VGG as backend

    # Args
        num_classes:
        arch: str. SSD architecture. Defaults to 'vgg_ssd_300'.
        input_shape:  tuple(int, int , int). Input shape of this network. Defaults to (300, 300, 3).
        weight_decay: float. Weight decay. Defaults to 5e-4.
        optimizer: list of dict, optional. Name of optimizer used for training.
        loss: list of dict, optional. Not Implemented Yet! List of loss objects to build for this model. Defaults to None.
        model_weights: str, optional. Path to the pretrained model weights. Defaults to None.
        summary: bool, optional. If true a summary of the model will be printed to stdout. Defaults to False.
        ratios: list of list of float, optional. List of anchor box ratios for SSD. Defaults to None.
        scales: list of float. List of scalings for anchor boxes. Defaults to None.
        class_score_threshold: float, optional. Class confidence threshold . Defaults to 0.6.
        nms_threshold: float. Min IoU for non-maximum suppression. Defaults to 0.45.
        max_boxes: int, optional. Max number of Boxes. Defaults to 200.

    # Returns
        A Keras model instance.

    # YAML Configuration
        ```
        model:
            ssd:
                num_classes: &num_classes 7
                input_shape:
                - 512
                - 512
                - 3
                summary: True
                optimizer:
                - Adam:
                    learning_rate: 0.001
                class_score_threshold: 0.6
                nms_threshold: 0.45
                max_boxes: 200
                arch: vgg_ssd_512
        ```
    # References
        - Single Shot Detector https://arxiv.org/abs/1512.02325
    """
    __defaults = {
        'ratios':
        {
            'vgg_ssd_300': [
                [1, 2, 0.5],
                [1, 2, 3, 0.5, 1 / 3],
                [1, 2, 3, 0.5, 1 / 3],
                [1, 2, 3, 0.5, 1 / 3],
                [1, 2, 0.5],
                [1, 2, 0.5]
            ],
            'vgg_ssd_512': [
                [1, 2, 0.5],
                [1, 2, 3, 0.5, 1 / 3],
                [1, 2, 3, 0.5, 1 / 3],
                [1, 2, 3, 0.5, 1 / 3],
                [1, 2, 3, 0.5, 1 / 3],
                [1, 2, 0.5],
                [1, 2, 0.5]
            ]
        },
        'scales':
        {
            'vgg_ssd_300': [0.1, 0.2, 0.375, 0.55, 0.725, 0.9, 1.075],
            'vgg_ssd_512': [0.07, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05]
        }
    }

    def __init__(self, num_classes, arch='vgg_ssd_300', input_shape=(300, 300, 3), weight_decay=5e-4,
                 optimizer=None, loss=None, model_weights=None, summary=False,
                 ratios=None, scales=None,
                 class_score_threshold=0.6, nms_threshold=0.45, max_boxes=200, **kwargs):
        self.weight_decay = weight_decay
        self.num_classes = num_classes
        self.max_boxes = max_boxes
        self.class_score_threshold = class_score_threshold
        self.nms_threshold = nms_threshold

        if ratios is None:
            if arch not in self.__defaults['ratios']:
                raise ValueError("No ratios found for arch %s" % arch)
            ratios = self.__defaults['ratios'][arch]

        if scales is None:
            if arch not in self.__defaults['scales']:
                raise ValueError("No ratios found for arch %s" % arch)
            scales = self.__defaults['scales'][arch]

        if 'vgg_' in arch:
            model, feature_maps_sizes, preprocess = VGG16_SSD(num_classes, ratios,
                                                              input_shape=input_shape,
                                                              ssd_300=arch == 'vgg_ssd_300')
        else:
            raise NotImplementedError(
                'Arch "%s" is not implemented yet' % arch)

        self.default_boxes = generate_default_boxes(
            scales, ratios, feature_maps_sizes)

        if summary:
            model.summary()

        if not loss:
            loss = {'SSDLoss': {
                'num_classes': num_classes,
                'neg_ratio': 3}
            }

        super().__init__(model, preprocessing=preprocess, optimizer=optimizer,
                         loss=loss, model_weights=model_weights, **kwargs)

    def training_step(self, record):
        if ('x_batch' not in record and 'y_batch' not in record):
            raise ValueError("Invalid data reader for SSD network")

        x_batch, y_batch = record['x_batch'], record['y_batch']

        target_labels = []
        target_boxes = []

        for i in range(x_batch.shape[0]):
            valid_indices = record['y_batch']['gt_labels'][i] != -1

            gt_label = tf.boolean_mask(
                record['y_batch']['gt_labels'][i], valid_indices)
            gt_box = tf.boolean_mask(
                record['y_batch']['gt_boxes'][i], valid_indices)

            target_label, target_box = compute_target(
                self.default_boxes, gt_box, gt_label)

            target_boxes.append(target_box)
            target_labels.append(target_label)

        target_boxes = tf.convert_to_tensor(target_boxes)
        target_labels = tf.convert_to_tensor(target_labels)

        if callable(self.preprocessing):
            x_batch = self.preprocessing(x_batch)

        confs, locs = self.model(x_batch)

        losses = {}
        conf_loss, loc_loss = self.losses[0](
            confs, locs, target_labels, target_boxes)

        loss = conf_loss + loc_loss
        l2_loss = [tf.nn.l2_loss(t) for t in self.model.trainable_variables]
        l2_loss = self.weight_decay * tf.math.reduce_sum(l2_loss)
        loss += l2_loss

        losses['conf_loss'] = conf_loss
        losses['loc_loss'] = loc_loss
        losses['total_l2_loss'] = loss

        confs = tf.math.softmax(confs, axis=-1)
        classes = tf.math.argmax(confs, axis=-1)
        scores = tf.math.reduce_max(confs, axis=-1)

        boxes = self.decode(locs)

        out_boxes = [[]] * confs.shape[0]
        out_classes = [[]] * confs.shape[0]
        out_scores = [[]] * confs.shape[0]

        # nms should be moved to a own function and changeable via model parameter
        for idx in range(confs.shape[0]):
            for c in range(1, self.num_classes):
                cls_scores = confs[idx, :, c]
                score_idx = cls_scores > self.class_score_threshold
                cls_boxes = boxes[idx][score_idx]
                cls_scores = cls_scores[score_idx]

                nms_idx = compute_nms(
                    cls_boxes, cls_scores, self.nms_threshold, self.max_boxes)
                cls_boxes = tf.gather(cls_boxes, nms_idx)
                cls_scores = tf.gather(cls_scores, nms_idx)
                cls_labels = [c] * cls_boxes.shape[0]

                out_boxes[idx].append(cls_boxes)
                out_classes[idx].extend(cls_labels)
                out_scores[idx].append(cls_scores)

            out_boxes[idx] = tf.concat(out_boxes[idx], axis=0)
            out_boxes[idx] = tf.clip_by_value(out_boxes[idx], 0.0, 1.0).numpy()
            out_scores[idx] = tf.concat(out_scores[idx], axis=0)
            out_classes[idx] = np.array(out_classes[idx])

        return losses, (out_classes, out_boxes, out_scores)

    def tape_step(self, tape, loss_values):
        if not self.optimizer:
            raise ValueError(
                "Optimizer not implemented in Model {}".format(self.__class__))

        grads = tape.gradient(
            loss_values['total_l2_loss'], self.model.trainable_weights)
        self.optimizer[0].apply_gradients(
            zip(grads, self.model.trainable_weights))

    def decode(self, locs):
        return decode(self.default_boxes, locs)

    def encode(self, boxes):
        return encode(self.default_boxes, boxes)

    def compute_target(self, gt_boxes, gt_labels):
        return compute_target(self.default_boxes, gt_boxes, gt_labels)
