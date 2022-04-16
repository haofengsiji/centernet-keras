import os
from datetime import datetime

import tensorflow as tf
from tensorflow import keras

from centernet import CenterNet
from data import VOCDataset
from losses import centernet_loss
from callbacks import VisCallback

gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def main():
    # load dataset
    train_dataset_raw = VOCDataset(data_path, input_shape, train_file,
                                   batch_size, True)
    train_dataset = train_dataset_raw.load_dataset()

    val_dataset_raw = VOCDataset(data_path, input_shape, train_file,
                                 batch_size, False)
    val_dataset = val_dataset_raw.load_dataset()

    vis_dataset_raw = VOCDataset(data_path, input_shape, train_file, 1, False)
    vis_dataset = vis_dataset_raw.load_dataset().repeat()

    steps_per_epoch = len(train_dataset_raw) // batch_size

    # callbacks
    logdir = os.path.join(log_path, datetime.now().strftime("%Y%m%d-%H%M%S"))
    tb_callback = keras.callbacks.TensorBoard(logdir, update_freq=100)
    ckpt_callback = keras.callbacks.ModelCheckpoint(
        filepath=logdir,
        save_best_only=True,
        monitor="val_loss",
        verbose=1,
    )
    vis_callback = VisCallback(vis_dataset,
                               class_names=train_dataset_raw.class_names,
                               logdir=logdir,
                               update_freq=100)

    # load model
    model = CenterNet(train_dataset_raw.class_names,
                      backbone_weights='imagenet',
                      freeze=freeze,
                      finetune=finetune)

    if ckpt_path:
        model.load_weights(ckpt_path)
        print(f"Loding pretrained weights from {ckpt_path} !")

    # model compile
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=lr,
        decay_steps=steps_per_epoch,
        decay_rate=0.94,
        staircase=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    model.compile(optimizer=optimizer, loss=centernet_loss, run_eagerly=False)

    model.fit(x=train_dataset,
              validation_data=val_dataset,
              epochs=epochs,
              callbacks=[tb_callback, ckpt_callback, vis_callback])


if __name__ == "__main__":
    model_path = ''
    input_shape = (512, 512)
    backbone = 'resnet50'

    epochs = 50
    batch_size = 2
    buffer_size = batch_size * 5
    lr = 1e-2
    freeze = True
    finetune = True or freeze
    ckpt_path = ""
    # ckpt_path =  "./logs/test/20220417-140619"

    data_path = "E:\github2\centernet-keras\VOCdevkit\VOC2007"
    train_file = "VOCdevkit\VOC2007\ImageSets\Main\\train.txt"
    val_file = "VOCdevkit\VOC2007\ImageSets\Main\\val.txt"

    log_path = "./logs/test"

    main()