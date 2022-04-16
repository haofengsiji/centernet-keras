import os
import math
from turtle import pos
import xml.etree.ElementTree as ET

from tqdm import tqdm
import cv2
import tensorflow as tf
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBoxesOnImage
ia.seed(0)


def preprocess_input(image):
    image = np.array(image, dtype=np.float32)
    return image / 127.5 - 1


def gaussian_radius(size, min_iou=0.7):
    h, w = size

    # intersection
    a1 = 1
    b1 = -(h + w)
    c1 = w * h * (1 - min_iou)
    sq1 = np.sqrt(b1**2 - 4 * a1 * c1)
    r1 = (-b1 + sq1) / 2

    # subset
    a2 = 4
    b2 = -2 * (h + w)
    c2 = w * h * (1 - min_iou)
    sq2 = np.sqrt(b2**2 - 4 * a2 * c2)
    r2 = (-b2 + sq2) / 2

    # superset
    a3 = 4 * min_iou
    b3 = 2 * min_iou * (h + w)
    c3 = (min_iou - 1) * h * w
    sq3 = np.sqrt(b3**2 - 4 * a3 * c3)
    r3 = (-b3 + sq3) / 2

    return min(r1, r2, r3)


def draw_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), diameter / 6)

    x, y = int(center[0]), int(center[1])

    h, w = heatmap.shape[0:2]

    left, right = min(x, radius), min(w - x, radius + 1)
    top, bottom = min(y, radius), min(h - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom,
                               radius - left:radius + right]

    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


def gaussian2D(shape, sigma=1):
    h, w = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-h:h + 1, -w:w + 1]

    map_2d = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    map_2d[map_2d < np.finfo(map_2d.dtype).eps * map_2d.max()] = 0

    return map_2d


class VOCDataset:
    ID2LABEL = [
        "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
        "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
        "pottedplant", "sheep", "sofa", "train", "tvmonitor"
    ]

    def __init__(self, path, 
                 input_shape, 
                 data_file,
                     batch_size,
                     training=False,
                     buffer_size=10,
                     class_names=None,):
        self.path = path
        self.input_shape = input_shape
        self.data_file = data_file
        self.batch_size = batch_size
        self.training = training
        self.buffer_size = buffer_size
        self.output_shape = (input_shape[0] // 4, input_shape[1] // 4)
        self.class_names = class_names if class_names else self.ID2LABEL
        self.num_classes = len(self.class_names)
        self.data_ids = self._get_data_id(data_file)
        
        if len(self.data_ids) % self.batch_size != 0:
            self.data_ids.extend(self.data_ids[:self.batch_size - (len(self.data_ids) % self.batch_size)]) 

        self.seq = None

    @staticmethod
    def _get_data_id(file):
        data_list = []
        with open(file) as f:
            for line in tqdm(f, desc="loading the context of data_file ..."):
                data_list.append(line.strip())
        return data_list

    def __len__(self):
        return len(self.data_ids)

    def __getitem__(self, idx):
        return self.getitem(idx)

    def getitem(self, idx):
        data_id = self.data_ids[idx]
        image_file = self.path + f"/JPEGImages/{data_id}.jpg"
        annot_file = self.path + f"/Annotations/{data_id}.xml"

        # fetch raw data
        if not os.path.exists(image_file):
            raise ValueError(f"{image_file} not exists")
        if not os.path.exists(annot_file):
            raise ValueError(f"{annot_file} not exists")

        image = cv2.imread(image_file)

        tree = ET.parse(open(annot_file))
        root = tree.getroot()

        boxes = []
        cls_ids = []
        for i, obj in enumerate(root.iter('object')):
            difficult = 0
            if obj.find('difficult') != None:
                difficult = obj.find('difficult').text
            cls = obj.find('name').text
            if cls not in self.class_names or int(difficult) == 1:
                continue
            cls_id = self.class_names.index(cls)
            xmlbox = obj.find('bndbox')
            b = [
                int(xmlbox.find('xmin').text),
                int(xmlbox.find('ymin').text),
                int(xmlbox.find('xmax').text),
                int(xmlbox.find('ymax').text)
            ]
            boxes.append(b)
            cls_ids.append(cls_id)

        boxes = np.array(boxes, np.float32)
        cls_ids = np.array(cls_ids, np.int32)

        # image(ih, iw) -> image(h, w)
        ih, iw, _ = image.shape
        h, w = self.input_shape

        if self.training:
            if self.seq is None:
                self.seq = iaa.Sequential([
                    iaa.Fliplr(0.5),
                    iaa.Resize({
                        "height": (0.7, 1.3),
                        "weight": (0.7, 1.3)
                    }),
                    iaa.Resize((.25, 2.)),
                    iaa.PadToFixedSize(width=w, height=h, pad_cval=128),
                    iaa.CropToFixedSize(width=w, height=h),
                    iaa.MultiplyHue((0.5, 1.5)),
                    iaa.MultiplySaturation((0.5, 1.5)),
                    iaa.MultiplyBrightness((0.5, 1.5)),
                    iaa.ClipCBAsToImagePlanes(),
                ])
        else:
            scale = min(w / iw, h / ih)
            self.seq = iaa.Sequential([
                iaa.Resize(scale),
                iaa.PadToFixedSize(width=w, height=h, pad_cval=128),
                iaa.CropToFixedSize(width=w, height=h, position="center"),
            ])

        bbs = BoundingBoxesOnImage.from_xyxy_array(boxes, (ih, iw))
        image, bbs = self.seq(image=image, bounding_boxes=bbs)
        boxes = bbs.to_xyxy_array()

        hms = np.zeros((*self.output_shape, self.num_classes), np.float32)
        whs = []
        regs = []
        reg_masks = []
        indices = []

        if len(boxes) != 0:
            boxes = np.array(boxes[:, :4], dtype=np.float32)
            boxes[:, [0, 2]] = np.clip(
                boxes[:, [0, 2]] / self.input_shape[1] * self.output_shape[1],
                0, self.output_shape[1] - 1)
            boxes[:, [1, 3]] = np.clip(
                boxes[:, [1, 3]] / self.input_shape[0] * self.output_shape[0],
                0, self.output_shape[0] - 1)

            for i in range(len(boxes)):
                bbox = boxes[i].copy()
                cls_id = cls_ids[i]

                h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
                if h > 0 and w > 0:
                    radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                    radius = max(0, int(radius))
                    #-------------------------------------------------#
                    #   计算真实框所属的特征点
                    #-------------------------------------------------#
                    ct = np.array([(bbox[0] + bbox[2]) / 2,
                                   (bbox[1] + bbox[3]) / 2],
                                  dtype=np.float32)
                    ct_int = ct.astype(np.int32)
                    #----------------------------#
                    #   绘制高斯热力图
                    #----------------------------#
                    hms[:, :, cls_id] = draw_gaussian(hms[:, :, cls_id],
                                                      ct_int, radius)
                    #---------------------------------------------------#
                    #   计算宽高真实值
                    #---------------------------------------------------#
                    whs.append([1. * w, 1. * h])
                    #---------------------------------------------------#
                    #   计算中心偏移量
                    #---------------------------------------------------#
                    regs.append(ct - ct_int)
                    #---------------------------------------------------#
                    #   将对应的mask设置为1，用于排除多余的0
                    #---------------------------------------------------#
                    reg_masks.append(1)
                    #---------------------------------------------------#
                    #   表示第ct_int[1]行的第ct_int[0]个。
                    #---------------------------------------------------#
                    indices.append(ct_int[1] * self.output_shape[0] +
                                   ct_int[0])
        if not whs:
            whs.append([0., 0.]) 
            regs.append([0., 0.])
            reg_masks.append(0)
            indices.append(0)

        image = preprocess_input(image)
        whs = np.array(whs, np.float32)
        regs = np.array(regs, np.float32)
        reg_masks = np.array(reg_masks, np.int32)
        indices = np.array(indices, np.int32)
        

        return image, boxes, cls_ids, hms, whs, regs, reg_masks, indices

    def generate(self):
        for i in range(len(self)):
            yield self[i]

    def load_dataset(self):
        dataset = tf.data.Dataset.from_generator(self.generate,
                                                 output_types=(
                                                     tf.float32,
                                                     tf.float32,
                                                     tf.uint8,
                                                     tf.float32,
                                                     tf.float32,
                                                     tf.float32,
                                                     tf.int32,
                                                     tf.int32,
                                                 ))

        if self.training:
            dataset = dataset.shuffle(self.buffer_size * self.batch_size)

        dataset = dataset.padded_batch(self.batch_size,
                                       padded_shapes=((
                                           *self.input_shape,
                                           3,
                                       ), (
                                           None,
                                           4,
                                       ), (None, ), (
                                           *self.output_shape,
                                           self.num_classes,
                                       ), (None, 2), (None, 2), (None, ),
                                                      (None, )))

        def orginize(image, boxes, cls_ids, hms, whs, regs, reg_masks,
                      indices):
            image.set_shape([self.batch_size, *self.input_shape, 3])
            boxes.set_shape([self.batch_size, None, 4])
            cls_ids.set_shape([self.batch_size, None])
            hms.set_shape([self.batch_size, *self.output_shape, self.num_classes])
            whs.set_shape([self.batch_size, None, 2])
            regs.set_shape([self.batch_size, None, 2])
            reg_masks.set_shape([self.batch_size, None])
            indices.set_shape([self.batch_size, None])

            return {
                "images": image,
            }, {
                "boxes": boxes,
                "cls_ids": cls_ids,
                "hms": hms,
                "whs": whs,
                "regs": regs,
                "reg_masks": reg_masks,
                "indices": indices
            }

        dataset = dataset.map(orginize)

        if self.training:
            dataset = dataset.prefetch(
                buffer_size=tf.data.experimental.AUTOTUNE)

        return dataset