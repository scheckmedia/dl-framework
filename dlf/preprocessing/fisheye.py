import tensorflow as tf

from dlf.core.registry import register_preprocessing_method
from dlf.core.preprocessing import PreprocessingMethod
from dlf.utils.helpers import random_apply

import time

import cv2
import numpy as np

try:
    from numba import jit
    has_numba = True
except:
    tf.get_logger().warning("Numba not found! Please install, using pip, to speed up this preprocessing")


@register_preprocessing_method('fisheye', 'fisheye_distortion', 'random_fisheye', 'radnom_fisheye_distortion')
class RandomFisheye(PreprocessingMethod):
    """Data augmentation method to create an interpolated fisheye distorted image of a perspective image

    # Aliases
        - fisheye
        - fisheye_distortion
        - random_fisheye
        - radnom_fisheye_distortion

    # Arguments
        fov: float, optional. Field of view for distortion. Defaults to 179.0.

    # Example

    ![](/computer-vision/dl-framework/img/preprocessing/fisheye.gif)


    # YAML Configuration

    ```yaml
        preprocess_list:
            fisheye:
    ```

    # Warning

    This module is very experimental and slow. You should install numba (`pip install numba`)
    otherwise it's realy unusable.

    # Note

    The bottleneck of this function is `remap = self.fisheye_transform(*image.shape[:-1])`.
    Maybe we should write it in cython or the are further optimizations available.

    """
    def __init__(self, fov=179.9):
        super().__init__()
        self.fov = np.clip(fov, 0, 179.9)

        if has_numba:
            self.warp = jit(self.warp)


    def _apply_transform(self, image, mask, boxes):
        if boxes is not None:
            raise NotImplementedError

        def wrap_py_fnc(image, mask):
            s = time.time()
            remap = self.fisheye_transform(*image.shape[:-1])

            s = time.time()
            image = cv2.remap(image.numpy().astype(np.uint8), remap, None, cv2.INTER_LINEAR)
            image = np.clip(image, 0, 255).astype(np.uint8)

            if mask is not None:
                s = time.time()
                mask = cv2.remap(mask.numpy().astype(np.uint8), remap, None, cv2.INTER_NEAREST, borderValue=(255,255,255))
                mask = np.clip(mask, 0, 255)

            return image.astype(np.float32), mask.astype(np.float32)

        image, mask = tf.py_function(wrap_py_fnc, [image, mask], [tf.float32, tf.float32])
        return image, mask, boxes


    def __call__(self, job):
        job.image, job.mask, job.boxes = random_apply(self._apply_transform, (job.image, job.mask, job.boxes), 0.5)
        return job

    def fisheye_transform(self, w, h):
        fov_half = 0.5 * self.fov * (np.pi / 180.0)
        max_factor = np.sin(fov_half)

        x, y = np.meshgrid(range(0, w), range(0, h))
        xy = np.dstack([x.flatten(),y.flatten()]).squeeze()

        map_xy = np.array([self.warp((x,y), max_factor, w, h) for x,y in xy], dtype=np.float32)
        map_xy = map_xy.reshape((h,w,2))
        return map_xy

    def warp(self, coords, max_factor, width, height):
        w = width - 1
        h = height - 1
        x,y = coords
        x = (x / w) * 2 - 1
        y = (y / h) * 2 - 1

        norm = np.sqrt(x**2 + y**2)

        if (norm > 1 or norm >= (2 * max_factor)):
            return np.inf, np.inf

        d = np.sqrt(((x * max_factor)**2 + (y * max_factor) ** 2))
        z = np.sqrt(1-d**2)

        phi = np.arctan2(y,x)
        r = np.arctan2(d,z) / np.pi

        u = r * np.cos(phi)
        v = r * np.sin(phi)

        u = (u + 0.5) * w
        v = (v + 0.5) * h
        return u, v