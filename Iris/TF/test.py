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

    classifier.train(
        input_fn=lambda: my_train_input_fn(train_x, train_y, 30),
        steps=1000
    )

    eval_result = classifier.evaluate(
        input_fn=lambda: my_eval_input_fn(test_x, test_y, 30)
    )

    print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

if __name__ == '__main__':
    tf.app.run(main)
