#!/usr/bin/env python3

# Created on: June-2024
# Created by: Kenny Le
# Created for: ICS4U
# This is the TensorFlow program

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras import layers

# Load CIFAR-10 dataset
(ds_train, ds_test), ds_info = tfds.load(
    "cifar10",
    split=["train", "test"],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

def normalize_img(image, label):
    """Normalizes images to [0, 1] range."""
    return tf.cast(image, tf.float32) / 255.0, label

AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 32

def augment(image, label):
    """Applies data augmentation to images."""
    if tf.random.uniform(()) < 0.1:
        image = tf.image.rgb_to_grayscale(image)
        image = tf.image.grayscale_to_rgb(image)

    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_flip_left_right(image)

    return image, label

# Setup for train dataset
ds_train = ds_train.map(normalize_img, num_parallel_calls=AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits["train"].num_examples)
ds_train = ds_train.map(augment, num_parallel_calls=AUTOTUNE)
ds_train = ds_train.batch(BATCH_SIZE)
ds_train = ds_train.prefetch(AUTOTUNE)

# Setup for test dataset
ds_test = ds_test.map(normalize_img, num_parallel_calls=AUTOTUNE)
ds_test = ds_test.batch(BATCH_SIZE)
ds_test = ds_test.prefetch(AUTOTUNE)

class_names = [
    "Airplane", "Automobile", "Bird", "Cat", "Deer",
    "Dog", "Frog", "Horse", "Ship", "Truck",
]

def get_model():
    model = keras.Sequential(
        [
            layers.Input((32, 32, 3)),
            layers.Conv2D(8, 3, padding="same", activation="relu"),
            layers.Conv2D(16, 3, padding="same", activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.1),
            layers.Dense(10),
        ]
    )
    return model

num_epochs = 1
loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
acc_metric = keras.metrics.SparseCategoricalAccuracy()

for learning_rate in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]:
    print(f"Starting training with learning rate: {learning_rate}")
    train_step = 0
    test_step = 0

    train_writer = tf.summary.create_file_writer("logs/train/" + str(learning_rate))
    test_writer = tf.summary.create_file_writer("logs/test/" + str(learning_rate))

    model = get_model()
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        
        # Iterate through training set
        for batch_idx, (x, y) in enumerate(ds_train):
            with tf.GradientTape() as tape:
                y_pred = model(x, training=True)
                loss = loss_fn(y, y_pred)

            gradients = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(gradients, model.trainable_weights))
            acc_metric.update_state(y, y_pred)

            with train_writer.as_default():
                tf.summary.scalar("Loss", loss, step=train_step)
                tf.summary.scalar("Accuracy", acc_metric.result(), step=train_step)
                train_step += 1

            if batch_idx % 100 == 0:
                print(f"Batch {batch_idx}, Loss: {loss.numpy()}, Accuracy: {acc_metric.result().numpy()}")

        # Reset accuracy metric
        acc_metric.reset_state()

        # Iterate through test set
        for batch_idx, (x, y) in enumerate(ds_test):
            y_pred = model(x, training=False)
            loss = loss_fn(y, y_pred)
            acc_metric.update_state(y, y_pred)

            with test_writer.as_default():
                tf.summary.scalar("Loss", loss, step=test_step)
                tf.summary.scalar("Accuracy", acc_metric.result(), step=test_step)
                test_step += 1

        print(f"Test Accuracy: {acc_metric.result().numpy()}")
        # Reset accuracy metric
        acc_metric.reset_state()

    print(f"Finished training with learning rate: {learning_rate}")

