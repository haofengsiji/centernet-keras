"""
Author:
    Zhengfang Xin, xinzhengfang@qq.com

Reference:
    https://github.com/bubbliiiing/centernet-tf2/blob/main/train.py
    

"""
import os
from datetime import datetime

import tensorflow as tf
from tensorflow import keras

from centernet import CenterNet
from data import VOCDataset
from losses import centernet_loss
from callbacks import VisCallback

def main():
    # load dataset
    voc_dataset = VOCDataset(
    "E:\github2\centernet_keras\VOCdevkit\VOC2007",
    input_shape)

    train_dataset = voc_dataset.load_dataset("E:\github2\centernet_keras\VOCdevkit\VOC2007\ImageSets\Main\\train.txt", batch_size, True)
    val_dataset = voc_dataset.load_dataset("E:\github2\centernet_keras\VOCdevkit\VOC2007\ImageSets\Main\\val.txt", 1, False)
    test_dataset = voc_dataset.load_dataset("E:\github2\centernet_keras\VOCdevkit\VOC2007\ImageSets\Main\\val.txt", 1, False).repeat()
    
    steps_per_epoch = len(voc_dataset) // batch_size


    # callbacks
    logdir = os.path.join(log_path, datetime.now().strftime("%Y%m%d-%H%M%S"))
    tb_callback = keras.callbacks.TensorBoard(logdir, update_freq=100)
    ckpt_callback = keras.callbacks.ModelCheckpoint(
        filepath=logdir,
        save_best_only=True,
        monitor="val_loss",
        verbose=1,
    )
    vis_callback = VisCallback( test_dataset, class_names=voc_dataset.class_names, logdir=logdir, update_freq=100)

    # load model
    model = CenterNet(voc_dataset.class_names)

    # model compile
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=lr,
        decay_steps=steps_per_epoch,
        decay_rate=0.94,
        staircase=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    model.compile(
        optimizer=optimizer,
        loss=centernet_loss,
        run_eagerly=False)
    
    model.fit(x=train_dataset,
              validation_data=val_dataset,
              epochs=1,
              callbacks=[tb_callback, ckpt_callback, vis_callback])


if __name__ == "__main__":
    model_path = ''
    input_shape = (256, 256)
    backbone = 'resnet50'

    epochs = 50
    batch_size = 2
    buffer_size = batch_size * 5
    lr = 1e-4

    train_file = "VOCdevkit\VOC2007\ImageSets\Main\\train.txt"
    val_file = "VOCdevkit\VOC2007\ImageSets\Main\\val.txt"

    log_path = "./logs/test"

    main()