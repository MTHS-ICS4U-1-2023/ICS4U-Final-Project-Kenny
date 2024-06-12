#!/usr/bin/env python3

# Created on: June-2024
# Created by: Kenny Le
# Created for: ICS4U
# This is the TensorFlow program

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import io
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import tensorflow_datasets as tfds

from tensorflow import keras
from tensorflow.keras import layers

from utils import plot_to_projector

(ds_train, ds_test), ds_info = tfds.load(
    "mnist",
    split=["train", "test"],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

def normalize_img(image, label):
    """Normalizes images"""
    return tf.cast(image, tf.float32) / 255.0, label


AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 64  # Reduced batch size

def augment(image, label):
    return image, label


# Setup for train dataset
print("Setting up train dataset...")
ds_train = ds_train.map(normalize_img, num_parallel_calls=AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits["train"].num_examples)
ds_train = ds_train.map(augment)
ds_train = ds_train.batch(BATCH_SIZE)
ds_train = ds_train.prefetch(AUTOTUNE)
print("Train dataset setup completed.")

# Setup for test Dataset
print("Setting up test dataset...")
ds_test = ds_test.map(normalize_img, num_parallel_calls=AUTOTUNE)
ds_test = ds_test.cache()
ds_test = ds_test.batch(BATCH_SIZE)
ds_test = ds_test.prefetch(AUTOTUNE)
print("Test dataset setup completed.")

class_names = [
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"
]

print("Fetching a batch of data for visualization...")
x_batch, y_batch = next(iter(ds_train))
print("Data fetched successfully.")

print("Creating visualization...")
plot_to_projector(x_batch, x_batch, y_batch, class_names, log_dir="proj")
print("Visualization created successfully.")
