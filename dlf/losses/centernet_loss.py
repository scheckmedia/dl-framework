import tensorflow as tf

from dlf.core.registry import register_loss


@register_loss('CenternetLoss', 'centernetloss', 'center_net_loss', 'centernet_loss')
class CenterNetLoss(tf.keras.losses.Loss):
    """Implementation of CenterNet loss

    # Aliases
        - CenternetLoss
        - centernetloss
        - center_net_loss
        - centernet_loss

    # Arguments
        loss_scale: int. Scaling factor for the loss
        loss_offset: float. Offset factor for the loss

    # YAML Configuration
    ```yaml
    loss:
        centernetloss:
            loss_scale: 0.1
            loss_offset: 1.0
    ```

    # References
        - [Objects as Points](https://arxiv.org/abs/1904.07850)
    """

    def __init__(self, loss_scale=0.1, loss_offset=1.0):
        super().__init__(name='centernet_loss')
        self.loss_scale = loss_scale
        self.loss_offset = loss_offset

    def __call__(self, heatmap, boxsize, offset, heatmap_gt, boxsize_gt, offset_gt, masks, indices):
        heatmap_loss = self.focal_loss(heatmap, heatmap_gt)
        boxsize_loss = self.loss_scale * \
            self.reg_l1_loss(boxsize, boxsize_gt, indices, masks)
        offset_loss = self.reg_l1_loss(offset, offset_gt, indices, masks)

        return heatmap_loss, boxsize_loss, offset_loss

    def focal_loss(self, hm_pred, hm_true):
        pos_mask = tf.cast(tf.equal(hm_true, 1), tf.float32)
        neg_mask = tf.cast(tf.less(hm_true, 1), tf.float32)
        neg_weights = tf.pow(1 - hm_true, 4)

        pos_loss = -tf.math.log(tf.clip_by_value(hm_pred, 1e-4,
                                                 1. - 1e-4)) * tf.pow(1 - hm_pred, 2) * pos_mask
        neg_loss = -tf.math.log(tf.clip_by_value(1 - hm_pred, 1e-4, 1. - 1e-4)
                                ) * tf.math.pow(hm_pred, 2) * neg_weights * neg_mask

        num_pos = tf.reduce_sum(pos_mask)
        pos_loss = tf.reduce_sum(pos_loss)
        neg_loss = tf.reduce_sum(neg_loss)

        cls_loss = tf.cond(tf.greater(num_pos, 0), lambda: (
            pos_loss + neg_loss) / num_pos, lambda: neg_loss)
        return cls_loss

    def reg_l1_loss(self, y_pred, y_true, indices, mask):
        b = tf.shape(y_pred)[0]
        k = tf.shape(indices)[1]
        c = tf.shape(y_pred)[-1]
        y_pred = tf.reshape(y_pred, (b, -1, c))
        indices = tf.cast(indices, tf.int32)
        y_pred = tf.gather(y_pred, indices, batch_dims=1)
        mask = tf.tile(tf.expand_dims(mask, axis=-1), (1, 1, 2))
        total_loss = tf.reduce_sum(tf.abs(y_true * mask - y_pred * mask))
        reg_loss = total_loss / (tf.reduce_sum(mask) + 1e-4)
        return reg_loss
