#!/usr/bin/env python3

# Created on: June-2024
# Created by: Kenny Le
# Created for: ICS4U
# This is the TensorFlow program

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pathlib  # pathlib is in standard library

batch_size = 2
img_height = 28
img_width = 28

# Directory containing your dataset
directory = "/home/ec2-user/environment/ICS4U/Final-Project/ICS4U-Final-Project-Kenny/TensorFlow18/data/mnist_images_only/"
ds_train = tf.data.Dataset.list_files(str(pathlib.Path(directory + "*.jpg")))


def process_path(file_path):
    # Read and decode image
    image = tf.io.read_file(file_path)
    image = tf.image.decode_jpeg(image, channels=1)
    image = tf.image.resize(image, [img_height, img_width])  # Resize to desired dimensions
    image = tf.cast(image, tf.float32) / 255.0  # Normalize image data
    
    # Extract label from the file path
    parts = tf.strings.split(file_path, os.path.sep)
    label = tf.strings.substr(parts[-1], pos=0, len=1)
    label = tf.strings.to_number(label, out_type=tf.int64)
    
    return image, label


# Map the process_path function to the dataset
ds_train = ds_train.map(process_path).batch(batch_size)

# Define the model
model = keras.Sequential([
    layers.Input((img_height, img_width, 1)),
    layers.Conv2D(16, 3, padding="same"),
    layers.Conv2D(32, 3, padding="same"),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(10),
])

# Compile the model
model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=[keras.losses.SparseCategoricalCrossentropy(from_logits=True)],
    metrics=["accuracy"],
)

# Train the model
model.fit(ds_train, epochs=10, verbose=2)
