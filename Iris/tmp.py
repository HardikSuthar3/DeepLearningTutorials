import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tf.keras


def load_data():
    # Load cifar10 data
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    # Normalize data
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    # Convert labels to one-hot
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)
    return x_train, y_train, x_test, y_test


def build_model():
    # Build model
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(
        32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    # add max pooling layer
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(10))
    return model


model = build_model()

# Compile model
model.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Load data
x_train, y_train, x_test, y_test = load_data()

# train model
model.fit(x_train, y_train, epochs=10, batch_size=32)

# Evaluate model
model.evaluate(x_test, y_test, batch_size=32, verbose=2)
