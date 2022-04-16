"""
Author:
    Zhengfang Xin, xinzhengfang@qq.com

Reference:
    https://www.delftstack.com/howto/python/opencv-object-detection/
    
"""

import tensorflow as tf
from tensorflow import keras
import cv2
from seaborn import color_palette
import numpy as np
np.random.seed(0)

from centernet import decode


class VisCallback(keras.callbacks.Callback):
    def __init__(self, dataset, class_names, logdir="./logs", conf=0.8, update_freq=100, mean=[0.40789655, 0.44719303, 0.47026116], std=[0.2886383, 0.27408165, 0.27809834]):
        super(VisCallback, self).__init__()
        self.dataset = iter(dataset)
        self.log_dir = logdir
        self.conf = conf
        self.update_freq = update_freq
        self.mean = mean
        self.std = std
        self.class_names = class_names
        self.colors = np.random.randint(0,
                                        255,
                                        size=(len(class_names), 3),
                                        dtype='uint8')
    
    def on_batch_end(self, batch, logs=None):
        if (batch + 1) % self.update_freq == 0:
            x, y = next(self.dataset)
            images = x["images"]
            boxes = y["boxes"]
            cls_ids = y["cls_ids"]
            hm_true = y["hms"]
            reg_masks = y["reg_masks"]
            
            hm_pred, wh_pred, reg_pred = self.model.predict(images)
            detections_pred = decode(hm_pred, wh_pred, reg_pred, self.model.max_objects)
            reg_masks = reg_masks
            images = images
            
            def img_vis(images, boxes, cls_ids, reg_masks, detections):
                images0 = images[0]
                boxes0 = boxes[0]
                cls_ids0 = cls_ids[0]
                detections0 = detections[0]

                images0 = (images0 * self.std + self.mean)*255
                images0 = images0.astype(np.uint8)
                
                # ground truth
                num_valid = int(np.sum(reg_masks[0]))
                for i in range(num_valid):
                    x1, y1, x2, y2 = boxes0[i].astype(np.int32) * 4
                    cls_id = cls_ids0[i].astype(np.int32)
                    color = [int(c) for c in self.colors[cls_id]]
                    cv2.rectangle(images0, (x1, y1), (x2, y2), color, 1)
                    text = "{}".format(self.class_names[cls_id])
                    cv2.putText(images0, text, (x1, y2 + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                                color, 1)
                    
                # pred
                boxes_pred = detections0[:, :4].astype(np.int32)
                conf_pred = detections0[:, 4].astype(np.float32)
                cls_ids_pred = detections0[:, 5].astype(np.uint8)
                num_valid = int(np.sum(conf_pred > 0.3))
                for i in range(num_valid):
                    x1, y1, x2, y2 = boxes_pred[i].astype(np.int32) * 4
                    cls_id = cls_ids_pred[i].astype(np.int32)
                    color = [int(c) for c in self.colors[cls_id]]
                    color[1] = 255
                    cv2.rectangle(images0, (x1, y1), (x2, y2), color, 1)
                    text = "{}: {:.4f}".format(self.class_names[cls_id], conf_pred[i])
                    cv2.putText(images0, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                                color, 1)
                
                images0 = cv2.cvtColor(images0, cv2.COLOR_BGR2RGB)    
                
                return images0[None, ...]
            
            hm_pred = tf.reduce_max(hm_pred, axis=-1)[ ..., None]
            hm_true = tf.reduce_max(hm_true, axis=-1)[ ..., None]
            images = tf.numpy_function(img_vis, [images, boxes, cls_ids, reg_masks, detections_pred, ], [tf.uint8])
            tf.summary.image("img_vis", images, step=batch, max_outputs=1)
            tf.summary.image("hm_pred", hm_pred, step=batch, max_outputs=1)
            tf.summary.image("hm_true", hm_true, step=batch, max_outputs=1)
                
                
                
                
                
                
                
                
                
            
            
            
    