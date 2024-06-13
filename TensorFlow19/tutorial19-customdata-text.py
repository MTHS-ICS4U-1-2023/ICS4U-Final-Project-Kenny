#!/usr/bin/env python3

# Created on: June-2024
# Created by: Kenny Le
# Created for: ICS4U
# This is the TensorFlow program

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
import pandas as pd
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras import layers
import pickle

# Using tf.keras.preprocessing.text.Tokenizer
from tensorflow.keras.preprocessing.text import Tokenizer

# Create a tokenizer instance
tokenizer = Tokenizer()

print("Loading datasets...")
english = tf.data.TextLineDataset("/home/ec2-user/environment/ICS4U/Final-Project/ICS4U-Final-Project-Kenny/TensorFlow19/english.csv")
swedish = tf.data.TextLineDataset("/home/ec2-user/environment/ICS4U/Final-Project/ICS4U-Final-Project-Kenny/TensorFlow19/swedish.csv")
dataset = tf.data.Dataset.zip((english, swedish))

print("Tokenizing datasets...")
for eng, swe in dataset.skip(1):
    eng_text = eng.numpy().decode("UTF-8")
    swe_text = swe.numpy().decode("UTF-8")
    tokenizer.fit_on_texts([eng_text, swe_text])
    print(tokenizer.texts_to_sequences([eng_text]))
    print(tokenizer.texts_to_sequences([swe_text]))

print("Creating example with multiple files...")
file_names = ["test_example1.csv", "test_example2.csv", "test_example3.csv"]
dataset = tf.data.TextLineDataset(file_names)

dataset1 = tf.data.TextLineDataset("/home/ec2-user/environment/ICS4U/Final-Project/ICS4U-Final-Project-Kenny/TensorFlow19/test_example1.csv").skip(1)
dataset2 = tf.data.TextLineDataset("/home/ec2-user/environment/ICS4U/Final-Project/ICS4U-Final-Project-Kenny/TensorFlow19/test_example2.csv").skip(1)
dataset3 = tf.data.TextLineDataset("/home/ec2-user/environment/ICS4U/Final-Project/ICS4U-Final-Project-Kenny/TensorFlow19/test_example3.csv").skip(1)

dataset = dataset1.concatenate(dataset2).concatenate(dataset3)

for line in dataset:
    print(line)

def filter_train(line):
    split_line = tf.strings.split(line, ",", maxsplit=4)
    dataset_belonging = split_line[1]  # train, test
    sentiment_category = split_line[2]  # pos, neg, unsup

    return (
        True
        if dataset_belonging == "train" and sentiment_category != "unsup"
        else False
    )

def filter_test(line):
    split_line = tf.strings.split(line, ",", maxsplit=4)
    dataset_belonging = split_line[1]  # train, test
    sentiment_category = split_line[2]  # pos, neg, unsup

    return (
        True if dataset_belonging == "test" and sentiment_category != "unsup" else False
    )

print("Filtering datasets...")
ds_train = tf.data.TextLineDataset("imdb.csv").filter(filter_train)
ds_test = tf.data.TextLineDataset("imdb.csv").filter(filter_test)

def build_vocabulary(ds_train, threshold=200):
    """ Build a vocabulary """
    frequencies = {}
    vocabulary = set()
    vocabulary.update(["sostoken"])
    vocabulary.update(["eostoken"])

    print("Building vocabulary...")
    for line in ds_train.skip(1):
        split_line = tf.strings.split(line, ",", maxsplit=4)
        review = split_line[4]
        tokenized_text = review.numpy().lower().split()

        for word in tokenized_text:
            if word not in frequencies:
                frequencies[word] = 1
            else:
                frequencies[word] += 1

            if frequencies[word] == threshold:
                vocabulary.update(tokenized_text)

    return vocabulary

print("Building vocabulary and saving it to file...")
vocabulary = build_vocabulary(ds_train)
vocab_file = open("vocabulary.obj", "wb")
pickle.dump(vocabulary, vocab_file)

print("Loading vocabulary from file...")
encoder = tfds.deprecated.text.TokenTextEncoder(
    list(vocabulary), oov_token="<UNK>", lowercase=True
)

def my_encoder(text_tensor, label):
    encoded_text = encoder.encode(text_tensor.numpy().decode('utf-8'))
    return encoded_text, label

def encode_map_fn(line):
    split_line = tf.strings.split(line, ",", maxsplit=4)
    label_str = split_line[2]  # neg, pos
    review = "sostoken " + split_line[4] + " eostoken"
    label = 1 if label_str == "pos" else 0

    (encoded_text, label) = tf.py_function(
        my_encoder, inp=[review, label], Tout=(tf.int64, tf.int32),
    )

    encoded_text.set_shape([None])
    label.set_shape([])
    return encoded_text, label

print("Preparing datasets for training...")
AUTOTUNE = tf.data.experimental.AUTOTUNE
ds_train = ds_train.map(encode_map_fn, num_parallel_calls=AUTOTUNE).cache()
ds_train = ds_train.shuffle(25000)
ds_train = ds_train.padded_batch(32, padded_shapes=([None], ()))

ds_test = ds_test.map(encode_map_fn)
ds_test = ds_test.padded_batch(32, padded_shapes=([None], ()))

print("Building model...")
model = keras.Sequential(
    [
        layers.Masking(mask_value=0),
        layers.Embedding(input_dim=len(vocabulary) + 2, output_dim=32,),
        layers.GlobalAveragePooling1D(),
        layers.Dense(64, activation="relu"),
        layers.Dense(1),
    ]
)

model.compile(
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(3e-4, clipnorm=1),
    metrics=["accuracy"],
)

print("Training model...")
model.fit(ds_train, epochs=15, verbose=2)

print("Evaluating model...")
model.evaluate(ds_test)
