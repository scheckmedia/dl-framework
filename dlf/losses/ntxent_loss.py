import tensorflow as tf
import numpy as np

from dlf.core.registry import register_loss


@register_loss('NTXentLoss', 'ntxentloss', 'ntx_ent_loss', 'ntexnt_loss')
class NTXentLoss(tf.keras.losses.Loss):
    """Implementation of NTXentLoss like used in SimCLR

    # Arguments
        batch_size: int. Used batch size
        temperature: float. Temperature to scale features

    # YAML Configuration
    ```yaml
    loss:
        NTXentLoss:
            batch_size: 16
            temperature: 0.5
    ```

    # References
        - [SimCLR](https://arxiv.org/pdf/2002.05709.pdf)
        - [Tensorflow implementation](https://github.com/google-research/simclr/blob/f3ca72f7efc085ad4abdb65f7a63459d9cfda78f/objective.py)
    """

    def __init__(self, batch_size, temperature=0.5):

        super().__init__(name='ntexnt_loss')
        self.temperature = temperature
        self.batch_size = batch_size
        self.LARGE_NUM = 1e9
        self.masks = tf.one_hot(tf.range(batch_size),
                                batch_size, dtype=np.float32)

        self.criterion = tf.keras.losses.CategoricalCrossentropy(
            from_logits=True, reduction=tf.keras.losses.Reduction.SUM)
        self.labels = tf.one_hot(
            tf.range(self.batch_size), self.batch_size * 2)

    def __call__(self, zis, zjs):

        logits_aa = tf.matmul(zis, zis, transpose_b=True) / self.temperature
        logits_aa = logits_aa - self.masks * self.LARGE_NUM
        logits_bb = tf.matmul(zjs, zjs, transpose_b=True) / self.temperature
        logits_bb = logits_bb - self.masks * self.LARGE_NUM

        logits_ab = tf.matmul(zis, zjs, transpose_b=True) / self.temperature
        logits_ba = tf.matmul(zjs, zis, transpose_b=True) / self.temperature

        loss_a = self.criterion(y_pred=tf.concat(
            [logits_ab, logits_aa], 1), y_true=self.labels)
        loss_b = self.criterion(y_pred=tf.concat(
            [logits_ba, logits_bb], 1), y_true=self.labels)
        loss = loss_a + loss_b

        return loss, (logits_ab, logits_ba)
