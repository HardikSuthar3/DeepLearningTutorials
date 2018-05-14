import tensorflow as tf
import pandas as pd
import numpy as np
import keras

TRAIN_URL = "http://download.tensorflow.org/data/iris_training.csv"
TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"

CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth',
                    'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']

batch_size = 30

def load_train_data():
    fileName = keras.utils.get_file(TRAIN_URL.split('/')[-1], origin=TRAIN_URL)

    train_data = pd.read_csv(filepath_or_buffer=fileName)

    y = train_data.pop("Species")
    x = train_data
    x = x.to_dict('list')

    return x, y

def load_test_data():
    fileName = keras.utils.get_file(TEST_URL.split('/')[-1], origin=TEST_URL)

    train_data = pd.read_csv(filepath_or_buffer=fileName)

    y = train_data.pop("Species")
    x = train_data
    x = x.to_dict('list')

    return x, y

def my_train_ip_fn(ds):
    ds = ds.shuffle(250).batch(batch_size=batch_size)
    return ds.make_one_shot_iterator().get_next()

def my_eval_ip_fn(ds):
    ds = ds.shuffle(250).batch(batch_size=batch_size)
    return ds.make_one_shot_iterator().get_next()

def main(argv):
    x, y = load_train_data()
    ds = tf.data.Dataset.from_tensor_slices((x, y))

    feature_column = []

    for name in x.keys():
        feature_column.append(tf.feature_column.numeric_column(key=name, dtype=tf.float32))

    model = tf.estimator.DNNClassifier(
        hidden_units=[10, 10],
        feature_columns=feature_column,
        model_dir="./models",
        n_classes=4,
        activation_fn=tf.nn.relu
    )

    model.train(
        input_fn=lambda: my_train_ip_fn(ds),
        steps=1000
    )

    x, y = load_test_data()
    ds = tf.data.Dataset.from_tensor_slices((x, y))

    eval_result = model.evaluate(
        input_fn=lambda: my_eval_ip_fn(ds)
    )

    print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

if __name__ == '__main__':
    tf.app.run(main)
