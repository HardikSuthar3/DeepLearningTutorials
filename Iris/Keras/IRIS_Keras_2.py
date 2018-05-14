import tensorflow as tf
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

TRAIN_URL = "http://download.tensorflow.org/data/iris_training.csv"
TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"

CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth',
                    'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']

batch_size = 30

def load_train_data():
    fileName = tf.keras.utils.get_file(TRAIN_URL.split('/')[-1], origin=TRAIN_URL)

    train_data = pd.read_csv(filepath_or_buffer=fileName)

    y = train_data.pop("Species")
    x = train_data
    x = x.to_dict('list')

    return x, y

def load_test_data():
    fileName = tf.keras.utils.get_file(TEST_URL.split('/')[-1], origin=TEST_URL)

    train_data = pd.read_csv(filepath_or_buffer=fileName)

    y = train_data.pop("Species")
    x = train_data
    x = x.to_dict('list')

    return x, y

def my_train_ip_fn(ds):
    ds = ds.batch(batch_size=batch_size)
    return ds.make_one_shot_iterator().get_next()

def my_eval_ip_fn(ds):
    ds = ds.batch(batch_size=batch_size)
    return ds.make_one_shot_iterator().get_next()

def main(argv):
    data = load_iris()
    train_x, test_x, train_y, test_y = train_test_split(data["data"], data["target"], train_size=0.8)

    train_y = tf.keras.utils.to_categorical(train_y, num_classes=4)
    test_y = tf.keras.utils.to_categorical(test_y, num_classes=4)

    ds = tf.data.Dataset.from_tensor_slices((train_x, train_y))

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(
        units=64,
        activation=tf.keras.activations.relu,
        input_shape=(4,)
    ))
    # model.add(tf.keras.layers.Dropout(0.1))
    model.add(tf.keras.layers.Dense(
        units=64,
        activation=tf.keras.activations.relu
    ))
    # model.add(tf.keras.layers.Dropout(0.1))
    model.add(tf.keras.layers.Dense(units=4, activation=tf.keras.activations.softmax))

    model.compile(
        optimizer=tf.keras.optimizers.SGD(),
        loss=tf.keras.losses.categorical_crossentropy,
        metrics=[tf.keras.metrics.binary_accuracy]
    )
    model.summary()
    iris_est = tf.keras.estimator.model_to_estimator(
        keras_model=model,
        model_dir="./models/"
    )

    iris_est.train(input_fn=lambda: my_train_ip_fn(ds), steps=1000)

    ds = tf.data.Dataset.from_tensor_slices((test_x, test_y))
    result = iris_est.evaluate(input_fn=lambda: my_eval_ip_fn(ds))
    print(result)
    print(ds.output_types)
    print(ds.output_shapes)

if __name__ == '__main__':
    tf.app.run(main)
