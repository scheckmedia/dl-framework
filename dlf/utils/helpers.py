import numpy as np
import sys
import tensorflow as tf
from . import string_int_label_map_pb2
from google.protobuf import text_format


def color_map(N=256, normalized=False, rgba=False):
    """Generates a colormap with N unique colors

    Keyword Arguments:
        N {int} -- size of color palette (default: {256})
        normalized {bool} -- defines whether the values are between 0 - 1.0 (normalized) or 0 - 255 (default: {False})

    Returns:
        [list] -- list with N colors
    """

    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3 if not rgba else 4), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7 - j)
            g = g | (bitget(c, 1) << 7 - j)
            b = b | (bitget(c, 2) << 7 - j)
            c = c >> 3

        if not rgba:
            cmap[i] = np.array([r, g, b])
        else:
            cmap[i] = np.array([r, g, b, 255])

    cmap = cmap / 255 if normalized else cmap
    return cmap


cmap = color_map()
cmap[255] = (255, 255, 255)
cmap_normalized = color_map(normalized=False, rgba=True)
cmap_normalized[255] = (1.0, 1.0, 1.0, 0.0)


def random_apply(fnc, data, p, seed=1):
    random = tf.random.uniform([1], minval=0, maxval=1)
    if isinstance(data, (list, tuple)):
        return tf.cond(random <= p, lambda: fnc(*data), lambda: data)

    return tf.cond(random <= p, lambda: fnc(data), lambda: data)


def parse_label_map(path, index_as_id=False):
    with tf.io.gfile.GFile(path, 'r') as fid:
        label_map_string = fid.read()
        label_map = string_int_label_map_pb2.StringIntLabelMap()
        try:
            text_format.Merge(label_map_string, label_map)
        except text_format.ParseError:
            label_map.ParseFromString(label_map_string)

        items = {}
        # pylint: disable=E1101
        for item in label_map.item:
            if item.display_name:
                items[item.id] = item.display_name
            else:
                items[item.id] = item.name
    return items


class RedirectOut():
    def __init__(self, out):
        super().__init__()
        self.out = out
        self.original = sys.stdout

    def __enter__(self):
        self.__fd = open(self.out, 'w')
        sys.stdout = self.__fd

    def __exit__(self, type, value, traceback):
        sys.stdout = self.original
        self.__fd.close()
