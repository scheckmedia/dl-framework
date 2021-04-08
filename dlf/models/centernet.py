import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model

from dlf.core.registry import register_model, get_active_experiment
from dlf.core.builder import build_model_wrapper
from dlf.core.model import ModelWrapper
from dlf.layers.centernet_detections import CenternetDetections
from dlf.utils.keypoint_utils import draw_gaussian, gaussian_radius


@register_model('CenterNet', 'center_net', 'centernet')
class CenterNet(ModelWrapper):
    """A implementation of CenterNet for object detection

    # Aliases
        - CenterNet
        - center_net
        - centernet

    # Arguments
        feature_extractor: dict. A feature extractor
        num_classes: int. Number of classes to detect
        score_threshold: float, optional. Minimum classification score to be a valid box
        pooling_nms: bool, optional. If true, max pooling is used as non maximum suppression. Defaults to False.
        dropout_rate: float, optional. Dropout rate if None no dropout is used. Defaults to None.
        max_objects: int, optional. max number of objects to detect. Defaults to 100.
        weight_decay: float, optional. Weight decay factor. Defaults to None.
        model_weights: str, optional. Path to the pretrained model weights. Defaults to None.
        summary: bool, optional. If true a summary of the model will be printed to stdout. Defaults to False.
        optimizer: list of dict, optional. Name of optimizer used for training.
        loss: list of dict, optional. List of loss objects to build for this model. Defaults to None.

    # YAML Configuration
        ```yaml
        model:
            centernet:
                feature_extractor:
                model_name: ResNet50
                input_shape:
                    - &width 512
                    - &height 512
                    - 3
                weights: "imagenet"
                num_classes: 6
                weight_decay: 0.0005
                summary: True
                loss:
                    centernetloss:

                optimizer:
                - Adam: # lr for simclr unsupervised
                    learning_rate:
                        PolynomialDecay:
                        initial_learning_rate: 0.001
                        decay_steps: 195312
                        end_learning_rate: 0.00001
        ```

        # References
        - [Objects as Points](https://arxiv.org/abs/1904.07850)
        - [Keras Implementation](https://github.com/xuannianz/keras-CenterNet)
    """

    def __init__(self, feature_extractor, num_classes=12, score_threshold=0.1, pooling_nms=True, dropout_rate=None, max_objects=100,
                 weight_decay=None, model_weights=None, summary=False, optimizer=None, loss=None, **kwargs):

        self.max_objects = max_objects
        self.exp = get_active_experiment()
        self.num_classes = num_classes# + 1 # + 1 because of 0 = background
        self.weight_decay = weight_decay

        feature_extractor_wrapper = build_model_wrapper(
            'feature_extractor', feature_extractor)
        feature_extractor = feature_extractor_wrapper.model

        num_filters = 256
        num_decoder_blocks = 3
        stride = 4
        self.output_size = np.array(
            feature_extractor.inputs[0].shape[1:3]) // stride
        x = feature_extractor.outputs[0]

        if dropout_rate is not None:
            x = tf.keras.layers.Dropout(rate=dropout_rate)(x)

        for i in range(num_decoder_blocks):
            x = tf.keras.layers.Conv2DTranspose(num_filters, (4, 4), strides=2, use_bias=False, padding='same',
                                                kernel_initializer='he_normal',
                                                name="encoder/block_{}/conv2d_transpose_1".format(i))(x)
            x = tf.keras.layers.BatchNormalization(
                name="encoder/block_{}/bn_1".format(i))(x)
            x = tf.keras.layers.ReLU(
                name="encoder/block_{}/relu_1".format(i))(x)

        # heat map header
        y1 = tf.keras.layers.Conv2D(num_filters, 3, padding='same',
                                    kernel_initializer='he_normal',
                                    name="heatmap_header/conv2d_1")(x)
        y1 = tf.keras.layers.BatchNormalization(name="heatmap_header/bn_1")(y1)
        y1 = tf.keras.layers.ReLU(name="heatmap_header/relu_1")(y1)
        y1 = tf.keras.layers.Conv2D(self.num_classes, 1, kernel_initializer='he_normal',
                                    name="heatmap_header/conv2d_2")(y1)
        y1 = tf.keras.layers.Activation(
            'sigmoid', name="heatmap_header/sigmoid_1")(y1)

        # box size header
        y2 = tf.keras.layers.Conv2D(num_filters, 3, padding='same',
                                    kernel_initializer='he_normal',
                                    name="boxsize_header/conv2d_1")(x)
        y2 = tf.keras.layers.BatchNormalization(name="boxsize_header/bn_1")(y2)
        y2 = tf.keras.layers.ReLU(name="boxsize_header/relu_1")(y2)
        y2 = tf.keras.layers.Conv2D(2, 1, kernel_initializer='he_normal',
                                    name="boxsize_header/conv2d_2")(y2)

        # reg header
        y3 = tf.keras.layers.Conv2D(num_filters, 3, padding='same',
                                    kernel_initializer='he_normal',
                                    name="offset_header/conv2d_1")(x)
        y3 = tf.keras.layers.BatchNormalization(
            name="offset_header/bn_1")(y3)
        y3 = tf.keras.layers.ReLU(name="offset_header/relu_1")(y3)
        y3 = tf.keras.layers.Conv2D(2, 1, kernel_initializer='he_normal',
                                    name="offset_header/conv2d_2")(y3)
        predictions = CenternetDetections(
            score_threshold=score_threshold,
            nms=pooling_nms,
            max_objects=self.max_objects)([y1, y2, y3])

        model = Model(inputs=feature_extractor.input,
                      outputs=[y1, y2, y3, predictions])

        if summary:
            model.summary()

        super().__init__(model, feature_extractor_wrapper.preprocessing, optimizer=optimizer,
                         loss=loss, model_weights=model_weights, **kwargs)

    def training_step(self, record):
        if ('x_batch' not in record and 'y_batch' not in record):
            raise ValueError("Invalid data reader for CenterNet")

        x_batch, y_batch = record['x_batch'], record['y_batch']

        if callable(self.preprocessing):
            x_batch = self.preprocessing(x_batch)

        heatmap, boxsize, offset, predictions = self.model(
            x_batch, training=True)
        heatmap_gt, boxsize_gt, offset_gt, masks, indices = self.encode(
            x_batch, y_batch)

        losses = {}
        heatmap_loss, boxsize_loss, offset_loss = self.losses[0](
            heatmap, boxsize, offset, heatmap_gt, boxsize_gt, offset_gt, masks, indices)

        total_loss = heatmap_loss + boxsize_loss + offset_loss

        if self.weight_decay is not None:
            l2_loss = [tf.nn.l2_loss(t)
                       for t in self.model.trainable_variables]
            l2_loss = self.weight_decay * tf.math.reduce_sum(l2_loss)
            total_loss += l2_loss

        y1, x1, y2, x2, out_scores, out_classes = tf.unstack(
            predictions, axis=2)

        out_boxes = tf.stack([y1, x1, y2, x2], axis=2)
        out_classes = tf.cast(out_classes, tf.int32)

        losses['heatmap'] = heatmap_loss
        losses['boxsize'] = boxsize_loss
        losses['offset'] = offset_loss
        losses['total'] = total_loss

        return losses, (out_classes, out_boxes, out_scores)

    def tape_step(self, tape, loss_values):
        if not self.optimizer:
            raise ValueError(
                "Optimizer not implemented in Model {}".format(self.__class__))

        grads = tape.gradient(
            loss_values['total'], self.model.trainable_weights)
        self.optimizer[0].apply_gradients(
            zip(grads, self.model.trainable_weights))

    # todo: optimize for graph mode!
    # @tf.function
    def encode(self, x_batch, y_batch):

        batch_hms = np.zeros((len(x_batch), self.output_size[0], self.output_size[1], self.num_classes),
                             dtype=np.float32)
        batch_boxsizes = np.zeros(
            (len(x_batch), self.max_objects, 2), dtype=np.float32)
        batch_offsets = np.zeros(
            (len(x_batch), self.max_objects, 2), dtype=np.float32)
        batch_offset_masks = np.zeros(
            (len(x_batch), self.max_objects), dtype=np.float32)
        batch_indices = np.zeros(
            (len(x_batch), self.max_objects), dtype=np.float32)

        batch_gt_boxes = y_batch['gt_boxes'].numpy()
        batch_gt_labels = y_batch['gt_labels'].numpy().astype(np.int32)

        # bottleneck! but numpy is faster than tf eager
        for b in range(len(x_batch)):
            x, gt_boxes, gt_labels = x_batch[b], batch_gt_boxes[b], batch_gt_labels[b]

            valid_indices = gt_labels != -1

            gt_labels = gt_labels[valid_indices]
            gt_boxes = gt_boxes[valid_indices]

            for i in range(len(gt_boxes)):
                box, cid = gt_boxes[i], gt_labels[i]
                y1, x1 = box[0] * self.output_size[0], box[1] * \
                    self.output_size[1]
                y2, x2 = box[2] * self.output_size[0], box[3] * \
                    self.output_size[1]
                h = y2 - y1
                w = x2 - x1

                # if h < 0 or w < 0:
                #     continue

                center = ((y1 + y2) * 0.5, (x1 + x2) * 0.5)
                center_int = np.array(center).astype(np.int32)

                h_ceil = np.ceil(h)
                w_ceil = np.ceil(w)
                radius = int(gaussian_radius(h_ceil, w_ceil))

                draw_gaussian(batch_hms[b, :, :, cid], center_int, radius)
                batch_boxsizes[b, i] = h, w
                batch_indices[b, i] = center_int[0] * \
                    self.output_size[1] + center_int[1]
                batch_offsets[b, i] = center - center_int
                batch_offset_masks[b, i] = 1

        return batch_hms, batch_boxsizes, batch_offsets, batch_offset_masks, batch_indices
