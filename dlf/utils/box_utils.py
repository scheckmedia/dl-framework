# borrowed from https://raw.githubusercontent.com/ChunML/ssd-tf2/master/box_utils.py
import tensorflow as tf
import itertools
import math


def compute_area(top_left, bot_right, clip=512.0):
    """ Compute area given top_left and bottom_right coordinates
    Args:
        top_left: tensor (num_boxes, 2)
        bot_right: tensor (num_boxes, 2)
    Returns:
        area: tensor (num_boxes,)
    """
    # top_left: N x 2
    # bot_right: N x 2
    hw = tf.clip_by_value(bot_right - top_left, 0.0, clip)
    area = hw[..., 0] * hw[..., 1]

    return area


def compute_iou(boxes_a, boxes_b, clip=512.0):
    """ Compute overlap between boxes_a and boxes_b
    Args:
        boxes_a: tensor (num_boxes_a, 4)
        boxes_b: tensor (num_boxes_b, 4)
    Returns:
        overlap: tensor (num_boxes_a, num_boxes_b)
    """
    # boxes_a => num_boxes_a, 1, 4
    boxes_a = tf.expand_dims(boxes_a, 1)

    # boxes_b => 1, num_boxes_b, 4
    boxes_b = tf.expand_dims(boxes_b, 0)
    top_left = tf.math.maximum(boxes_a[..., :2], boxes_b[..., :2])
    bot_right = tf.math.minimum(boxes_a[..., 2:], boxes_b[..., 2:])

    overlap_area = compute_area(top_left, bot_right, clip)
    area_a = compute_area(boxes_a[..., :2], boxes_a[..., 2:], clip)
    area_b = compute_area(boxes_b[..., :2], boxes_b[..., 2:], clip)

    overlap = overlap_area / (area_a + area_b - overlap_area)

    return overlap


def compute_target(default_boxes, gt_boxes, gt_labels, iou_threshold=0.5):
    """ Compute regression and classification targets
    Args:
        default_boxes: tensor (num_default, 4)
                       of format (cy, cx, h, w)
        gt_boxes: tensor (num_gt, 4)
                  of format (ymin, xmin, ymax, xmax)
        gt_labels: tensor (num_gt,)
    Returns:
        gt_confs: classification targets, tensor (num_default,)
        gt_locs: regression targets, tensor (num_default, 4)
    """
    # Convert default boxes to format (ymin, xmin, ymax, xmax)
    # in order to compute overlap with gt boxes
    transformed_default_boxes = transform_center_to_corner(default_boxes)
    iou = compute_iou(transformed_default_boxes, gt_boxes)

    best_gt_iou = tf.math.reduce_max(iou, 1)
    best_gt_idx = tf.math.argmax(iou, 1)

    best_default_iou = tf.math.reduce_max(iou, 0)
    best_default_idx = tf.math.argmax(iou, 0)

    best_gt_idx = tf.tensor_scatter_nd_update(
        best_gt_idx,
        tf.expand_dims(best_default_idx, 1),
        tf.range(best_default_idx.shape[0], dtype=tf.int64))

    # Normal way: use a for loop
    # for gt_idx, default_idx in enumerate(best_default_idx):
    #     best_gt_idx = tf.tensor_scatter_nd_update(
    #         best_gt_idx,
    #         tf.expand_dims([default_idx], 1),
    #         [gt_idx])

    best_gt_iou = tf.tensor_scatter_nd_update(
        best_gt_iou,
        tf.expand_dims(best_default_idx, 1),
        tf.ones_like(best_default_idx, dtype=tf.float32))

    gt_confs = tf.gather(gt_labels, best_gt_idx)
    gt_confs = tf.where(
        tf.less(best_gt_iou, iou_threshold),
        tf.zeros_like(gt_confs),
        gt_confs)

    gt_boxes = tf.gather(gt_boxes, best_gt_idx)
    gt_locs = encode(default_boxes, gt_boxes)

    return gt_confs, gt_locs


def encode(default_boxes, boxes, variance=[0.1, 0.2]):
    """ Compute regression values
    Args:
        default_boxes: tensor (num_default, 4)
                       of format (cy, cx, h, w)
        boxes: tensor (num_default, 4)
               of format (ymin, xmin, ymax, xmax)
        variance: variance for center point and size
    Returns:
        locs: regression values, tensor (num_default, 4)
    """
    # Convert boxes to (cy, cx, h, w) format
    transformed_boxes = transform_corner_to_center(boxes)

    locs = tf.concat([
        (transformed_boxes[..., :2] - default_boxes[:, :2]
         ) / (default_boxes[:, 2:] * variance[0]),
        tf.math.log(transformed_boxes[..., 2:] / default_boxes[:, 2:]) / variance[1]],
        axis=-1)

    return locs


def decode(default_boxes, locs, variance=[0.1, 0.2]):
    """ Decode regression values back to coordinates
    Args:
        default_boxes: tensor (num_default, 4)
                       of format (cy, cx, h, w)
        locs: tensor (batch_size, num_default, 4)
              of format (cy, cx, h, w)
        variance: variance for center point and size
    Returns:
        boxes: tensor (num_default, 4)
               of format (ymin, xmin, ymax, xmax)
    """
    locs = tf.concat([
        locs[..., :2] * variance[0] *
        default_boxes[:, 2:] + default_boxes[:, :2],
        tf.math.exp(locs[..., 2:] * variance[1]) * default_boxes[:, 2:]], axis=-1)

    boxes = transform_center_to_corner(locs)

    return boxes


def transform_corner_to_center(boxes):
    """ Transform boxes of format (ymin, xmin, ymax, xmax)
        to format (cy, cx, h, w)
    Args:
        boxes: tensor (num_boxes, 4)
               of format (ymin, xmin, ymax, xmax)
    Returns:
        boxes: tensor (num_boxes, 4)
               of format (cy, cx, h, w)
    """
    center_box = tf.concat([
        (boxes[..., :2] + boxes[..., 2:]) / 2,
        boxes[..., 2:] - boxes[..., :2]], axis=-1)

    return center_box


def transform_center_to_corner(boxes):
    """ Transform boxes of format (cy, cx, h, w)
        to format (ymin, xmin, ymax, xmax)
    Args:
        boxes: tensor (num_boxes, 4)
               of format (cy, cx, h, w)
    Returns:
        boxes: tensor (num_boxes, 4)
               of format (ymin, xmin, ymax, xmax)
    """
    corner_box = tf.concat([
        boxes[..., :2] - boxes[..., 2:] / 2,
        boxes[..., :2] + boxes[..., 2:] / 2], axis=-1)

    return corner_box


def compute_nms(boxes, scores, nms_threshold, limit=200):
    """ Perform Non Maximum Suppression algorithm
        to eliminate boxes with high overlap

    # Args:
        boxes: tensor (num_boxes, 4)
               of format (ymin, xmin, ymax, xmax)
        scores: tensor (num_boxes,)
        nms_threshold: NMS threshold
        limit: maximum number of boxes to keep

    # Returns:
        idx: indices of kept boxes
    """
    if boxes.shape[0] == 0:
        return tf.constant([], dtype=tf.int32)
    selected = [0]
    idx = tf.argsort(scores, direction='DESCENDING')
    idx = idx[:limit]
    boxes = tf.gather(boxes, idx)

    iou = compute_iou(boxes, boxes)

    while True:
        row = iou[selected[-1]]
        next_indices = row <= nms_threshold
        # iou[:, ~next_indices] = 1.0
        iou = tf.where(
            tf.expand_dims(tf.math.logical_not(next_indices), 0),
            tf.ones_like(iou, dtype=tf.float32),
            iou)

        if not tf.math.reduce_any(next_indices):
            break

        selected.append(tf.argsort(
            tf.dtypes.cast(next_indices, tf.int32), direction='DESCENDING')[0].numpy())

    return tf.gather(idx, selected)


def generate_default_boxes(scales, ratios, feature_map_sizes):
    """ Generate default boxes for all feature maps
    # Args:
        scales: boxes' size relative to image's size
        ratios: box ratios used in each feature maps
        feature_map_sizes: sizes of feature maps
    # Returns:
        default_boxes: tensor of shape (num_default, 4)
                       with format (cy, cx, h, w)
    """
    default_boxes = []
    for m, fm_size in enumerate(feature_map_sizes):
        for i, j in itertools.product(range(fm_size), repeat=2):
            cx = (j + 0.5) / fm_size
            cy = (i + 0.5) / fm_size

            default_boxes.append([
                cy,
                cx,
                math.sqrt(scales[m] * scales[m + 1]),
                math.sqrt(scales[m] * scales[m + 1])
            ])

            for ratio in ratios[m]:
                r = math.sqrt(ratio)
                default_boxes.append([
                    cy,
                    cx,
                    scales[m] / r,
                    scales[m] * r
                ])

    default_boxes = tf.constant(default_boxes)
    default_boxes = tf.clip_by_value(default_boxes, 0.0, 1.0)

    return default_boxes
