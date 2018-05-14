import tensorflow as tf
from iris import iris_data
import pandas as pd
import numpy as np


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

def main(unused_argv):
    (train_x, train_y), (test_x, test_y) = iris_data.load_data()

    my_feature_columns = []

    for Key in train_x.keys():
        my_feature_columns.append(tf.feature_column.numeric_column(Key))

    classifier = tf.estimator.DNNClassifier(
        hidden_units=[10, 10],
        feature_columns=my_feature_columns,
        n_classes=30
    )

    train_ip_fn = tf.estimator.inputs.numpy_input_fn(
        x=dict(train_x),
        y=train_y, batch_size=30,
        num_epochs=None,
        shuffle=False
    )
    classifier.train(
        input_fn=train_ip_fn,
        steps=10000
    )
    # classifier.train(
    #     input_fn=lambda: iris_data.train_input_fn(train_x, train_y, 30),
    #     steps=1000
    # )

    eval_ip_fn = tf.estimator.inputs.numpy_input_fn(
        x=dict(test_x),
        y=test_y,
        batch_size=30,
        shuffle=False,
        num_epochs=None
    )
    eval_result = classifier.evaluate(
        input_fn=eval_ip_fn
    )

    # eval_result = classifier.evaluate(
    #     input_fn=lambda: iris_data.eval_input_fn(test_x, test_y, 30)
    # )

    print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
