#!/usr/bin/env python3

# Created on: June-2024
# Created by: Kenny Le
# Created for: ICS4U
# This is the TensorFlow program

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers

# Creating a summary writer for TensorBoard logs
writer = tf.summary.create_file_writer("logs/graph_vis")

# Function to be traced by TensorBoard
@tf.function
def my_func(x, y):
    return tf.nn.relu(tf.matmul(x, y))

# Generate random tensors
x = tf.random.uniform((3, 3))
y = tf.random.uniform((3, 3))

print("Before function call")

# Enable tracing and profiling
tf.summary.trace_on(graph=True, profiler=True, profiler_outdir="logs/graph_vis")
out = my_func(x, y)
tf.summary.trace_export(
    name="function_trace", step=0
)

print("After function call")
