"""
Author:
    Zhengfang Xin, xinzhengfang@qq.com

Reference:
    https://github.com/bubbliiiing/centernet-tf2/blob/main/nets/centernet.py
    

"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import cv2

np.random.seed(0)

from resnet import ResNet50, centernet_head
from losses import centernet_loss

mean = [0.40789655, 0.44719303, 0.47026116]
std = [0.2886383, 0.27408165, 0.27809834]


def nms(heat, kernel=3):
    """cool NMS implementation!!!
    
    """
    hmax = layers.MaxPool2D(pool_size=kernel, strides=1, padding='same')(heat)
    heat = tf.where(tf.equal(hmax, heat), heat, tf.zeros_like(heat))
    return heat


def topk(hm, max_objects=100):
    """get topk boxes from heatmap

    Args:
        hm (batch_size, h, w, num_classes): heatmap
        max_objects (int, optional): maximum boxes, Defaults to 100.

    Returns:
        scores (batch_size, max_objects), tf.float32: confidence in the existence of boxes
        indices (batch_size, max_objects), tf.int32: indices (y*w + x) in the center of boxes
        class_ids (batch_size, max_objects), tf.int32: category of boxes
        xs (batch_size, max_objects), tf.int32: x in indices
        ys (batch_size, max_obejcts), tf.int32: y in indices
    """
    hm = nms(hm)
    b, h, w, c = hm.shape
    hm = tf.reshape(hm, (b, -1))  # (b, h*w*c)

    scores, indices = tf.math.top_k(hm, k=max_objects, sorted=True)  # (b, k)

    class_ids = indices % c
    xs = indices // c % w
    ys = indices // c // w
    indices = ys * w + xs
    return scores, indices, class_ids, xs, ys


def decode(hm, wh, reg, max_objects=100):
    """decode centernet output to boxes result

    Args:
        hm (b, h, w, c): heatmap
        wh (b, h, w, 2): boxes' width and hight
        reg (b, h, w, 2): boxes' center offset
        max_objects (int, optional): maximum output boxes. Defaults to 100.
    
    Returns:

    """
    scores, indices, class_ids, xs, ys = topk(hm, max_objects=max_objects)
    b = tf.shape(hm)[0]

    # (b, h*w, 2)
    reg = tf.reshape(reg, [b, -1, 2])
    wh = tf.reshape(wh, [b, -1, 2])
    length = tf.shape(wh)[1]

    # (b, k)
    batch_idx = tf.expand_dims(tf.range(0, b), 1)
    batch_idx = tf.tile(batch_idx, (1, max_objects))
    # full_indices (b*k) = b*(h*w) + indices
    full_indices = tf.reshape(batch_idx, [-1]) * tf.cast(
        length, tf.int32) + tf.reshape(indices, [-1])

    # index topk value: (b*(h*m), 2) -> (b*k, 2) -> (b, k, 2)
    # indices(indices): (b*k, 1)
    # reg/wh(shape): (b*(h*m), 2)
    # topk_reg/topk_wh(updates): (b*k, 2)
    topk_reg = tf.gather(tf.reshape(reg, [-1, 2]), full_indices[..., None])
    topk_reg = tf.reshape(topk_reg, [b, -1, 2])

    topk_wh = tf.gather(tf.reshape(wh, [-1, 2]), full_indices[..., None])
    topk_wh = tf.reshape(topk_wh, [b, -1, 2])

    # fine tune boxes' center
    # (b, k)
    topk_cx = tf.cast(xs, tf.float32) + topk_reg[..., 0]
    topk_cy = tf.cast(ys, tf.float32) + topk_reg[..., 1]

    # get top left location (x1, y1) and bottom right location (x2, y2)
    # (b, k)
    topk_x1, topk_y1 = topk_cx - topk_wh[..., 0] / 2, topk_cy - topk_wh[...,
                                                                        1] / 2
    topk_x2, topk_y2 = topk_cx + topk_wh[..., 0] / 2, topk_cy + topk_wh[...,
                                                                        1] / 2

    class_ids = tf.cast(class_ids, tf.float32)

    # detections (b, k, 6)
    detections = tf.stack(
        [topk_x1, topk_y1, topk_x2, topk_y2, scores, class_ids], axis=-1)

    return detections


def img_vis(images, boxes, cls_ids, reg_masks, detections, colors, class_names):
    images0 = images[0]
    boxes0 = boxes[0]
    cls_ids0 = cls_ids[0]
    detections0 = detections[0]

    images0 = (images0 * std + mean)*255
    images0 = images0.astype(np.uint8)
    
    # ground truth
    num_valid = int(np.sum(reg_masks[0]))
    for i in range(num_valid):
        x1, y1, x2, y2 = boxes0[i].astype(np.int32) * 4
        cls_id = cls_ids0[i].astype(np.int32)
        color = [int(c) for c in colors[cls_id]]
        cv2.rectangle(images0, (x1, y1), (x2, y2), color, 1)
        text = "{}".format(class_names[cls_id])
        cv2.putText(images0, text, (x1, y2 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    color, 1)
        
    # pred
    boxes_pred = detections0[:, :4].astype(np.int32)
    conf_pred = detections0[:, 4].astype(np.float32)
    cls_ids_pred = detections0[:, 5].astype(np.uint8)
    num_valid = int(np.sum(conf_pred > 0.3))
    for i in range(num_valid):
        x1, y1, x2, y2 = boxes_pred[i].astype(np.int32) * 4
        cls_id = cls_ids_pred[i].astype(np.int32)
        color = [int(c) for c in colors[cls_id]]
        color[1] = 255
        cv2.rectangle(images0, (x1, y1), (x2, y2), color, 2)
        text = "{}: {:.4f}".format(class_names[cls_id], conf_pred[i])
        cv2.putText(images0, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    color, 2)
    
    images0 = cv2.cvtColor(images0, cv2.COLOR_BGR2RGB)    
    
    return images0


def centernet(input_shape,
              num_classes,
              backbone="resnet50",
              backbone_weights=None):

    if backbone == "resnet50":
        backbone_model = ResNet50(input_shape, weights=backbone_weights)
        C5 = backbone_model.get_layer("activation_48").output

    y1, y2, y3 = centernet_head(C5, num_classes)

    model = keras.Model(backbone_model.inputs, [y1, y2, y3], name="centernet")
    return model


class CenterNet(keras.Model):

    def __init__(self,
                 class_name,
                 max_objects=100,
                 backbone="resnet50",
                 backbone_weights=None,
                 freeze=False,
                 **kwargs):
        super(CenterNet, self).__init__(**kwargs)
        self.class_name = class_name
        self.num_classes = len(class_name)
        self.max_objects = max_objects
        self.freeze = freeze
        self.backbone_weights = backbone_weights
        self.backbone = backbone

        if self.backbone == 'resnet50':
            self.freeze_layer = 175

    def build(self, input_shape):
        self.model = centernet(input_shape[1:], self.num_classes,
                               self.backbone, self.backbone_weights)

        if self.freeze:
            for i in range(self.freeze_layer):
                self.model.layers[i].trainable = False
            tf.print('Freeze the first {} layers of total {} layers.'.format(
                self.freeze_layer, len(self.model.layers)))

        return super().build(input_shape)

    def call(self, inputs, training=None, mask=None):
        hm, wh, reg = self.model(inputs, training)
        return hm, wh, reg

    def train_step(self, data):
        x, y = data
        images = x["images"]
        hm_true = y["hms"]
        wh_true = y["whs"]
        reg_true = y["regs"]
        reg_masks = y["reg_masks"]
        indices = y["indices"]

        with tf.GradientTape() as tape:
            hm_pred, wh_pred, reg_pred = self(images, training=True)
            hm_loss, wh_loss, reg_loss = centernet_loss(
                (hm_true, wh_true, reg_true, reg_masks, indices),
                (hm_pred, wh_pred, reg_pred))
            loss = sum(self.losses) + hm_loss + wh_loss + reg_loss

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Conpute metrics
        #TODO
        # Update metrics
        output = {m.name: m.result() for m in self.metrics}
        output.update({
            "loss": loss,
            "hm_loss": hm_loss,
            "wh_loss": wh_loss,
            "reg_loss": reg_loss
        })

        return output

    def test_step(self, data):
        # Unpack the data
        x, y = data
        images = x["images"]
        hm_true = y["hms"]
        wh_true = y["whs"]
        reg_true = y["regs"]
        reg_masks = y["reg_masks"]
        indices = y["indices"]

        # Compute predictions
        hm_pred, wh_pred, reg_pred = self(images, training=False)
        # Updates the metrics tracking the loss
        hm_loss, wh_loss, reg_loss = centernet_loss(
            (hm_true, wh_true, reg_true, reg_masks, indices),
            (hm_pred, wh_pred, reg_pred))
        loss = sum(self.losses) + hm_loss + wh_loss + reg_loss
        # Conpute metrics
        #TODO
        # Update metrics
        output = {m.name: m.result() for m in self.metrics}
        output.update({
            "loss": loss,
            "hm_loss": hm_loss,
            "wh_loss": wh_loss,
            "reg_loss": reg_loss
        })
        return output

    def summary(self, line_length=None, positions=None, print_fn=None):
        return self.model.summary(line_length, positions, print_fn)
