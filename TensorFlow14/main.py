#!/usr/bin/env python3

# Created on: June-2024
# Created by: Kenny Le
# Created for: ICS4U
# This is the TensorFlow program

import os
import matplotlib.pyplot

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_datasets as tfds

# Load MNIST dataset
(ds_train, ds_test), ds_info = tfds.load(
    "mnist",
    split=["train", "test"],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

# Normalize images
def normalize_img(image, label):
    """Normalizes images to the range [0, 1]"""
    return tf.cast(image, tf.float32) / 255.0, label

AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 128

# Setup for training dataset
ds_train = ds_train.map(normalize_img, num_parallel_calls=AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits["train"].num_examples)
ds_train = ds_train.batch(BATCH_SIZE)
ds_train = ds_train.prefetch(AUTOTUNE)

# Setup for test dataset
ds_test = ds_test.map(normalize_img, num_parallel_calls=AUTOTUNE)
ds_test = ds_test.batch(BATCH_SIZE)
ds_test = ds_test.prefetch(AUTOTUNE)

# Model definition
model = keras.Sequential(
    [
        keras.Input((28, 28, 1)),
        layers.Conv2D(32, 3, activation="relu"),
        layers.Flatten(),
        layers.Dense(10, activation="softmax"),
    ]
)

# ModelCheckpoint callback with correct file path
save_callback = keras.callbacks.ModelCheckpoint(
    "checkpoint/model.weights.h5", save_weights_only=True, monitor="accuracy", save_best_only=False,
)

# Learning rate scheduler
lr_scheduler = keras.callbacks.ReduceLROnPlateau(
    monitor="loss", factor=0.1, patience=3, mode="max", verbose=1
)

# Custom callback to stop training early based on accuracy
class OurOwnCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs.get("accuracy") > 0.7:
            print("Accuracy over 70%, quitting training")
            self.model.stop_training = True

# Model compilation
model.compile(
    optimizer=keras.optimizers.Adam(0.01),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"],
)

# Model training
model.fit(
    ds_train,
    epochs=10,
    callbacks=[save_callback, lr_scheduler, OurOwnCallback()],
    verbose=2,
)

# Model evaluation
model.evaluate(ds_test, verbose=2)