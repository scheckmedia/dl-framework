# source: https://github.com/xuannianz/keras-CenterNet/blob/39cb123a94d7774490df28e637240de03577f912/generators/utils.py#L105

# todo: @tf.function

import numpy as np


def gaussian_radius(height, width, min_overlap=0.7):
    """Calculates a Gaussian radius for a rectangle/bound box

    # Arguments
        height: float. height of bounding box
        width: float. width of bounding box
        min_overlap: float. overlap factor. Defaults to 0.7.

    # Returns
        [type]. [description]
    """
    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    return min(r1, r2, r3)

# todo: @tf.function


def draw_gaussian(heatmap, center, radius, k=1):
    """draws a Gaussian heatmap for a point with a given radius

    # Arguments
        heatmap: np.ndarray. existing heatmap
        center: np.ndarray. position to draw the the point
        radius: float. radius of headmap point
        k: int. [description]. Defaults to 1.

    # Returns
         np.ndarray. new heatmap containing the point
    """
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    y, x = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom,
                               radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


def gaussian2D(shape, sigma=1):
    """Creates a Gaussian heatmap point of a given shape

    # Arguments
        shape: [type]. [description]
        sigma: int. [description]. Defaults to 1.

    # Returns
        [type]. [description]
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h
