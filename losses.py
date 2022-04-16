"""
Author:
    Zhengfang Xin, xinzhengfang@qq.com

Reference:
    https://github.com/bubbliiiing/centernet-tf2/blob/main/nets/centernet_training.py

"""
import tensorflow as tf
from tensorflow import keras

def focal_loss(hm_pred, hm_true):
    """focal loss
    
    p_t = 1 - p
    y_pos = -(1 - p)^gama*log(p)
    p -> ground_truth(1), loss: -log(p) -> samller(0), weight: (1 - p)^gama smaller(0^gama)

    Args:
        hm_pred (tf.float32) 
        hm_true (tf.float32)

    Returns:
        scale: loss value
    """
    
    pos_mask = tf.cast(tf.equal(hm_true, 1), tf.float32)
    
    neg_mask = tf.cast(tf.less(hm_true, 1), tf.float32)
    neg_weights = tf.pow(1 - hm_true, 4)

    pos_loss = -tf.math.log(tf.clip_by_value(hm_pred, 1e-6, 1.)) * tf.pow(1 - hm_pred, 2) * pos_mask
    neg_loss = -tf.math.log(tf.clip_by_value(1 - hm_pred, 1e-6, 1.)) * tf.pow(hm_pred, 2) * neg_weights * neg_mask

    num_pos = tf.reduce_sum(pos_mask)
    pos_loss = tf.reduce_sum(pos_loss) 
    neg_loss = tf.reduce_sum(neg_loss)

    # normalization
    cls_loss = tf.cond(tf.greater(num_pos, 0), lambda: (pos_loss + neg_loss) / num_pos, lambda: neg_loss)
    
    return cls_loss


def reg_l1_loss(y_pred, y_true, indices, mask):
    """l1 loss for the regression task

    Args:
        y_pred (batch_size, h, w, 2)
        y_true (batch_size, max_objects, 2)
        indices (batch_size, max_objects): the indice(i*w + j) of hxw
        mask (batch_size, max_objects): 1: existent 0: non-existent

    Returns:
        scale: loss value
    """
    
    b, c = tf.shape(y_pred)[0], tf.shape(y_pred)[-1]
    k = tf.shape(indices)[1]

    y_pred = tf.reshape(y_pred, (b, -1, c))
    length = tf.shape(y_pred)[1]
    indices = tf.cast(indices, tf.int32)

    batch_idx = tf.expand_dims(tf.range(0, b), 1)
    batch_idx = tf.tile(batch_idx, (1, k))
    full_indices = (tf.reshape(batch_idx, [-1]) * tf.cast(length, tf.int32) +
                    tf.reshape(indices, [-1]))

    y_pred = tf.gather(tf.reshape(y_pred, [-1,c]),full_indices)
    y_pred = tf.reshape(y_pred, [b, -1, c])

    mask = tf.tile(tf.expand_dims(mask, axis=-1), (1, 1, 2))
    total_loss = tf.reduce_sum(tf.abs(y_true * mask - y_pred * mask))
    reg_loss = total_loss / (tf.reduce_sum(mask) + 1e-4)
    return reg_loss


def centernet_loss(y_true, y_pred):
    """

    Args:
        hm_true (batch_size, h, w, num_classes): the ground truth of the heatmap
        wh_true (batch_size, max_objects, 2): the ground truth of boxes' width and height
        reg_true    (batch_size, max_objects, 2): the ground truth of boxes' center offset
        reg_masks   (batch_size, max_objects): 1: box object 0: No box object
        indices (batch_size, max_objects): the indice(i*w + j) of the box center location for hxw
        hm_pred (batch_size, h, w, num_classes): the prediction for the heatmap
        wh_pred (batch_size, h, w, 2): the prediction for boxes' width and height
        reg_pred    (batch_size, h, w, 2): the prediction for boxes' center offset
    Returns:
        scale: centernet loss value
    """
    hm_true, wh_true, reg_true, reg_masks, indices = y_true
    reg_masks = tf.cast(reg_masks, tf.float32)
    hm_pred, wh_pred, reg_pred = y_pred
    hm_loss = focal_loss(hm_pred, hm_true)
    wh_loss = 0.1 * reg_l1_loss(wh_pred, wh_true, indices, reg_masks)
    reg_loss = reg_l1_loss(reg_pred, reg_true, indices, reg_masks)
    return hm_loss, wh_loss, reg_loss