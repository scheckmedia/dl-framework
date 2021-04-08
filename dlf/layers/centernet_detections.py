import tensorflow as tf


class CenternetDetections(tf.keras.layers.Layer):
    """Layer that converts centernet outputs to bounding boxes, classes and scores

        # Arguments
            max_objects: int. maximum number of objects to detect. Defaults to 100.
            score_threshold: float. threshold to be a valid bounding box. Defaults to 0.1.
            nms: bool. use max pooling non maximum suppression. Defaults to True.
            normalized: bool. Scale bounding boxes between 0 and 1. Defaults to True.

        # References
        - [based on Keras implementation](https://github.com/xuannianz/keras-CenterNet/blob/39cb123a94d7774490df28e637240de03577f912/models/resnet.py)
        """

    def __init__(self, max_objects=100, score_threshold=0.1, nms=True, normalized=True, **kwargs):
        self.max_objects = max_objects
        self.score_threshold = score_threshold
        self.use_nms = nms
        self.normalized = normalized
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [tf.keras.layers.InputSpec(shape=input_shape)]
        self.input_height, self.input_width, self.num_classes = self.num_classes = input_shape[
            0][1:]
        super().build(input_shape)

    def call(self, x):
        heatmaps, boxsizes, offsets = x
        scores, indices, class_ids, xs, ys = self.topk(
            heatmaps, max_objects=self.max_objects)

        b = tf.shape(heatmaps)[0]
        h = tf.cast(tf.shape(heatmaps)[1], tf.float32)
        w = tf.cast(tf.shape(heatmaps)[2], tf.float32)
        # (b, h * w, 2)
        offsets = tf.reshape(offsets, (b, -1, tf.shape(offsets)[-1]))
        # (b, h * w, 2)
        boxsizes = tf.reshape(boxsizes, (b, -1, tf.shape(boxsizes)[-1]))
        # (b, k, 2)
        topk_offsets = tf.gather(offsets, indices, batch_dims=1)
        # (b, k, 2)
        topk_boxsizes = tf.cast(
            tf.gather(boxsizes, indices, batch_dims=1), tf.float32)
        topk_cy = tf.cast(tf.expand_dims(ys, axis=-1),
                          tf.float32) + topk_offsets[..., 0:1]
        topk_cx = tf.cast(tf.expand_dims(xs, axis=-1),
                          tf.float32) + topk_offsets[..., 1:2]
        scores = tf.expand_dims(scores, axis=-1)
        class_ids = tf.cast(tf.expand_dims(class_ids, axis=-1), tf.float32)
        topk_y1 = topk_cy - topk_boxsizes[..., 0:1] / 2
        topk_y2 = topk_cy + topk_boxsizes[..., 0:1] / 2
        topk_x1 = topk_cx - topk_boxsizes[..., 1:2] / 2
        topk_x2 = topk_cx + topk_boxsizes[..., 1:2] / 2

        topk_y1 = tf.clip_by_value(topk_y1, 0, h)
        topk_y2 = tf.clip_by_value(topk_y2, 0, h)
        topk_x1 = tf.clip_by_value(topk_x1, 0, w)
        topk_x2 = tf.clip_by_value(topk_x2, 0, w)

        if self.normalized:
            topk_y1 /= h
            topk_y2 /= h
            topk_x1 /= w
            topk_x2 /= w

        # (b, k, 6)
        detections = tf.concat(
            [topk_y1, topk_x1, topk_y2, topk_x2, scores, class_ids], axis=-1)
        detections = tf.map_fn(lambda x: self.filter_scores(x[0], self.score_threshold, self.max_objects), elems=[detections], dtype=tf.float32)
        return detections

    @staticmethod
    def filter_scores(batch_item, score_threshold, max_objects):
        batch_item_detections = tf.boolean_mask(batch_item, tf.greater(batch_item[:, 4], score_threshold))
        num_detections = tf.shape(batch_item_detections)[0]
        num_pad = tf.maximum(max_objects - num_detections, 0)
        batch_item_detections = tf.pad(tensor=batch_item_detections, paddings=[
            [0, num_pad],
            [0, 0]],
            mode='CONSTANT',
            constant_values=0.0)

        return batch_item_detections

    def nms(self, heat, kernel=3):
        hmax = tf.nn.max_pool2d(heat, (kernel, kernel),
                                strides=1, padding='SAME')
        heat = tf.where(tf.equal(hmax, heat), heat, tf.zeros_like(heat))
        return heat

    def topk(self, heatmaps, max_objects=100):
        if self.use_nms:
            heatmaps = self.nms(heatmaps)

        # (b, h * w * c)
        b, h, w, c = tf.shape(heatmaps)[0], tf.shape(
            heatmaps)[1], tf.shape(heatmaps)[2], tf.shape(heatmaps)[3]
        heatmaps = tf.reshape(heatmaps, (b, -1))
        # (b, k), (b, k)
        scores, indices = tf.nn.top_k(heatmaps, k=max_objects)
        class_ids = indices % c
        xs = indices // c % w
        ys = indices // c // w
        indices = ys * w + xs
        return scores, indices, class_ids, xs, ys

    def get_config(self):
        config = {
            'max_objects': self.max_objects,
            'score_threshold': self.score_threshold,
            'use_nms': self.use_nms,
            'normalized': self.normalized
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
