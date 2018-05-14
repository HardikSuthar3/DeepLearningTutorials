import tensorflow as tf
from iris import iris_data
import pandas as pd
import numpy as np
from tensorflow.python.keras import layers, activations, losses, metrics, optimizers
from tensorflow.python.keras import models
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

def my_train_input_fn(features, labels, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    dataset = dataset.repeat().batch(batch_size)

    return dataset.make_one_shot_iterator().get_next()

def my_eval_input_fn(features, labels, batch_size):
    features = dict(features)

    if labels is None:
        dataset = tf.data.Dataset.from_tensor_slices(features)
    else:
        dataset = tf.data.Dataset.from_tensor_slices((features, labels))

    dataset = dataset.batch(batch_size)

    return dataset.make_one_shot_iterator().get_next()

def main(args):
    # (train_x, train_y), (test_x, test_y) = iris_data.load_data()
    data = load_iris()
    # print(type(data))
    # print(data.keys())
    train_x, test_x, train_y, test_y = train_test_split(data["data"], data["target"], train_size=0.7)

    train_y = tf.keras.utils.to_categorical(train_y, num_classes=4)
    test_y = tf.keras.utils.to_categorical(test_y, num_classes=4)

    model = models.Sequential()
    model.add(layers.Dense(units=32, activation=activations.relu, input_shape=(4,)))
    model.add(layers.Dense(units=32, activation=activations.relu))
    model.add(layers.Dense(units=4, activation=activations.softmax))

    model.compile(
        optimizer=optimizers.SGD(),
        loss=losses.categorical_crossentropy,
        metrics=[metrics.categorical_accuracy]
    )
    model.summary()

    iris_est = tf.keras.estimator.model_to_estimator(model, model_dir="./models/")

    iris_est.train(
        input_fn=lambda: my_train_input_fn(train_x, train_y, 30),
        steps=100
    )

    result = iris_est.evaluate(
        input_fn=lambda: my_eval_input_fn(test_x, test_y, 30)
    )

    print(result)
    # model.fit(
    #     x=train_x,
    #     y=train_y,
    #     batch_size=30,
    #     epochs=100,
    #     shuffle=True
    # )
    #
    # result = model.evaluate(x=test_x, y=test_y)
    #
    # print(result[1])

if __name__ == '__main__':
    tf.app.run(main)
