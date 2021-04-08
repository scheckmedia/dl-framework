import tensorflow as tf

from dlf.core.registry import register_preprocessing_method
from dlf.core.preprocessing import PreprocessingMethod
from dlf.utils.helpers import random_apply

import cv2
import numpy as np

@register_preprocessing_method('transform', '6dof', 'Transform', 'random_transform', 'RandomTransform')
class RandomTransform(PreprocessingMethod):
    """Data augmentation method that randomly transforms an image with 6 DoF

    # Aliases
        - transform
        - 6dof
        - Transform
        - random_transform
        - RandomTransform

    # Arguments
        min_pitch: float, optional.  Minimum ptich angle. Defaults to -60.0.
        max_pitch: float, optional.  Maximum ptich angle. Defaults to 60.0.
        min_roll: float, optional.  Minimum roll angle. Defaults to -60.0.
        max_roll: float, optional.  Maximum roll angle. Defaults to 60.0.
        min_yaw: float, optional.  Minimum yaw angle. Defaults to -60.0.
        max_yaw: float, optional.  Maximum yaw angle. Defaults to 60.0.
        min_translate_x: float, optional.  Minimum x translation postion (1.0 means 100% of width). Defaults to -1.0.
        max_translate_x: float, optional.  Maximum x translation postion (1.0 means 100% of width). Defaults to 1.0.
        min_translate_y: float, optional.  Minimum y translation postion (1.0 means 100% of height). Defaults to -1.0.
        max_translate_y: float, optional.  Maximum y translation postion (1.0 means 100% of height). Defaults to 1.0.
        min_translate_z: float, optional.  Minimum z translation postion (1.0 means 100% of height). Defaults to -1.0.
        max_translate_z: float, optional.  Maximum z translation postion (1.0 means 100% of height). Defaults to 1.0.
        min_scale_x: float, optional.  Minimum x scale. Defaults to 1.0.
        max_scale_x: float, optional.  Maximum x scale. Defaults to 1.0.
        min_scale_y: float, optional.  Minimum y scale. Defaults to 1.0.
        max_scale_y: float, optional.  Maximum y scale. Defaults to 1.0.
        min_shear_x: float, optional.  Minimum x shear. Defaults to -5.0.
        max_shear_x: float, optional.  Maximum x shear. Defaults to 5.0.
        min_shear_y: float, optional.  Minimum y shear. Defaults to -5.0.
        max_shear_y: float, optional.  Maximum y shear. Defaults to 5.0.
        min_shear_z: float, optional.  Minimum z shear. Defaults to -5.0.
        max_shear_z: float, optional.  Maximum z shear. Defaults to 5.0.

    # Example

    ![](/computer-vision/dl-framework/img/preprocessing/transform.gif)

    # YAML Configuration

    ```yaml
        preprocess_list:
            transform:
    ```


    # References:
        - https://towardsdatascience.com/how-to-transform-a-2d-image-into-a-3d-space-5fc2306e3d36
    """

    def __init__(self,
        min_pitch = -60.0,
        max_pitch = 60.0,
        min_roll = -60.0,
        max_roll = 60.0,
        min_yaw = -60.0,
        max_yaw = 60.0,

        min_translate_x = -1.0,
        max_translate_x = 1.0,
        min_translate_y = -1.0,
        max_translate_y = 1.0,
        min_translate_z = -1.0,
        max_translate_z = 1.0,

        min_scale_x = 1.0,
        max_scale_x = 1.0,
        min_scale_y = 1.0,
        max_scale_y = 1.0,

        min_shear_x = -5.0,
        max_shear_x = 5.0,
        min_shear_y = -5.0,
        max_shear_y = 5.0,
        min_shear_z = -5.0,
        max_shear_z = 5.0,

    ):
        super().__init__()

        self.min_pitch = min_pitch
        self.max_pitch = max_pitch
        self.min_roll = min_roll
        self.max_roll = max_roll
        self.min_yaw = min_yaw
        self.max_yaw = max_yaw

        self.min_translate_x = min_translate_x
        self.max_translate_x = max_translate_x
        self.min_translate_y = min_translate_y
        self.max_translate_y = max_translate_y
        self.min_translate_z = min_translate_z
        self.max_translate_z = max_translate_z

        self.min_scale_x = min_scale_x
        self.max_scale_x = max_scale_x
        self.min_scale_y = min_scale_y
        self.max_scale_y = max_scale_y

        self.min_shear_x = min_shear_x
        self.max_shear_x = max_shear_x
        self.min_shear_y = min_shear_y
        self.max_shear_y = max_shear_y
        self.min_shear_z = min_shear_z
        self.max_shear_z = max_shear_z

    @tf.function
    def _apply_transform(self, image, mask, boxes):

        if boxes is not None:
            raise NotImplementedError

        def wrap_py_fnc(image, mask):
            w, h = image.shape[1], image.shape[0]

            min_max_pairs = [
                (self.min_pitch, self.max_pitch),
                (self.min_roll, self.max_roll),
                (self.min_yaw, self.max_yaw),

                (self.min_translate_x, self.max_translate_x),
                (self.min_translate_y, self.max_translate_y),
                (self.min_translate_z, self.max_translate_z),

                (self.min_scale_x, self.max_scale_x),
                (self.min_scale_y, self.max_scale_y),

                (self.min_shear_x, self.max_shear_x),
                (self.min_shear_y, self.max_shear_y),
                (self.min_shear_z, self.max_shear_z),
            ]

            random_factors = [np.random.uniform(p[0], p[1]) for p in min_max_pairs]

            rotation = tuple(random_factors[0:3])
            translation = tuple(random_factors[3:6])
            scale = tuple(random_factors[6:8]) + (1.0,)
            shear = tuple(random_factors[8:])

            M = self.get_matrix(w, h, rotation, translation, scale, shear)
            image = cv2.warpPerspective(image.numpy().astype(np.uint8), M, (w, h), None, cv2.INTER_LINEAR)

            if mask is not None:
                mask = cv2.warpPerspective(mask.numpy().astype(np.uint8), M, (w, h), None, cv2.INTER_NEAREST, borderValue=(255,255,255))

            return image.astype(np.float32), mask.astype(np.float32)

        image, mask = tf.py_function(wrap_py_fnc, [image, mask], [tf.float32, tf.float32])
        return image, mask, boxes


    def __call__(self, job):
        job.image, job.mask, job.boxes = random_apply(self._apply_transform, (job.image, job.mask, job.boxes), 0.5)
        return job

    def get_matrix(self, w, h,
              rotation=(0, 0, 0),
              translation=(0, 0, 0),
              scaling=(1, 1, 1),
              shearing=(0, 0, 0)):

        # get the values on each axis
        t_x, t_y, t_z = translation
        r_x, r_y, r_z = rotation
        sc_x, sc_y, sc_z = scaling
        sh_x, sh_y, sh_z = shearing

        # convert degree angles to rad
        theta_rx = np.deg2rad(r_x)
        theta_ry = np.deg2rad(r_y)
        theta_rz = np.deg2rad(r_z)
        theta_shx = np.deg2rad(sh_x)
        theta_shy = np.deg2rad(sh_y)
        theta_shz = np.deg2rad(sh_z)

        # compute its diagonal
        diag = (h ** 2 + w ** 2) ** 0.5
        # compute the focal length
        f = diag
        if np.sin(theta_rz) != 0:
            f /= 2 * np.sin(theta_rz)


        # set the image from cartesian to projective dimension
        H_M = np.array([[1, 0, -w / 2],
                        [0, 1, -h / 2],
                        [0, 0,      1],
                        [0, 0,      1]])
        # set the image projective to carrtesian dimension
        Hp_M = np.array([[f, 0, w / 2, 0],
                        [0, f, h / 2, 0],
                        [0, 0,     1, 0]])

        Identity = np.array([[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])

        # adjust the translation on z
        t_z = (f - t_z) / sc_z ** 2
        # translation matrix to translate the image
        T_M = np.array([[1, 0, 0, t_x],
                        [0, 1, 0, t_y],
                        [0, 0, 1, t_z],
                        [0, 0, 0,  1]])

        # calculate cos and sin of angles
        sin_rx, cos_rx = np.sin(theta_rx), np.cos(theta_rx)
        sin_ry, cos_ry = np.sin(theta_ry), np.cos(theta_ry)
        sin_rz, cos_rz = np.sin(theta_rz), np.cos(theta_rz)
        # get the rotation matrix on x axis
        R_Mx = np.array([[1,      0,       0, 0],
                        [0, cos_rx, -sin_rx, 0],
                        [0, sin_rx,  cos_rx, 0],
                        [0,      0,       0, 1]])
        # get the rotation matrix on y axis
        R_My = np.array([[cos_ry, 0, -sin_ry, 0],
                        [     0, 1,       0, 0],
                        [sin_ry, 0,  cos_ry, 0],
                        [     0, 0,       0, 1]])
        # get the rotation matrix on z axis
        R_Mz = np.array([[cos_rz, -sin_rz, 0, 0],
                        [sin_rz,  cos_rz, 0, 0],
                        [     0,       0, 1, 0],
                        [     0,       0, 0, 1]])
        # compute the full rotation matrix
        R_M = np.dot(np.dot(R_Mx, R_My), R_Mz)

        # get the scaling matrix
        Sc_M = np.array([[sc_x,     0,    0, 0],
                        [   0,  sc_y,    0, 0],
                        [   0,     0, sc_z, 0],
                        [   0,     0,    0, 1]])


        # get the tan of angles
        tan_shx = np.tan(theta_shx)
        tan_shy = np.tan(theta_shy)
        tan_shz = np.tan(theta_shz)
        # get the shearing matrix on x axis
        Sh_Mx = np.array([[      1, 0, 0, 0],
                        [tan_shy, 1, 0, 0],
                        [tan_shz, 0, 1, 0],
                        [      0, 0, 0, 1]])
        # get the shearing matrix on y axis
        Sh_My = np.array([[1, tan_shx, 0, 0],
                        [0,       1, 0, 0],
                        [0, tan_shz, 1, 0],
                        [0,       0, 0, 1]])
        # get the shearing matrix on z axis
        Sh_Mz = np.array([[1, 0, tan_shx, 0],
                        [0, 1, tan_shy, 0],
                        [0, 0,       1, 0],
                        [0, 0,       0, 1]])
        # compute the full shearing matrix
        Sh_M = np.dot(np.dot(Sh_Mx, Sh_My), Sh_Mz)

        # compute the full transform matrix
        M = H_M
        M = np.dot(Sc_M, M)
        M = np.dot(R_M,  M)
        M = np.dot(T_M,  M)
        M = np.dot(Sh_M, M)
        M = np.dot(Hp_M, M)
        # apply the transformation
        return M


