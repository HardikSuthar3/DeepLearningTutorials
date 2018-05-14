import tensorflow as tf
from tensorflow.python.keras import layers, activations, losses, metrics, models, optimizers
from tensorflow.python.keras import datasets
import keras.backend as K

tf.logging.set_verbosity(tf.logging.INFO)

# Load data
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

batch_size = 128
num_classes = 10
epochs = 12
img_rows, img_cols = 28, 28

# Load data
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
y_train = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=num_classes)

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

def get_cnn_model():
    # cnn_model.add(layers.Conv2D(
    #     filters=64,
    #     strides=[2, 2],
    #     kernel_size=[2, 2],
    #     padding="same",
    #     activation=activations.relu,
    #     input_shape=input_shape
    # ))
    # cnn_model.add(layers.MaxPooling2D(pool_size=[2, 2], strides=2, padding="same"))
    #
    # cnn_model.add(layers.Conv2D(
    #     filters=32,
    #     strides=[2, 2],
    #     kernel_size=[2, 2],
    #     padding="same",
    #     activation=activations.relu
    # ))
    # cnn_model.add(layers.MaxPooling2D(pool_size=[2, 2], strides=2, padding="same"))
    # cnn_model.add(layers.Dropout(0.25))
    # cnn_model.add(layers.Flatten())
    # cnn_model.add(layers.Dropout(0.25))
    # cnn_model.add(layers.Dense(
    #     units=128,
    #     activation=activations.relu,
    # ))
    # cnn_model.add(layers.Dropout(0.25))
    # cnn_model.add(layers.Dense(num_classes, tf.nn.softmax))
    #
    # cnn_model.compile(
    #     optimizer=optimizers.Adagrad(),
    #     loss=losses.categorical_crossentropy,
    #     metrics=[metrics.top_k_categorical_accuracy]
    # )
    input = layers.Input(shape=input_shape, dtype=tf.float32, name="x")

    c1 = layers.Conv2D(
        filters=32,
        kernel_size=[2, 2], strides=2,
        padding="same",
        activation=activations.relu
    )(input)
    mp1 = layers.MaxPooling2D(
        strides=2, padding="same", pool_size=[2, 2]
    )(c1)

    c2 = layers.Conv2D(
        filters=64,
        kernel_size=[2, 2], strides=2,
        padding="same",
        activation=activations.relu
    )(mp1)
    mp2 = layers.MaxPooling2D(
        strides=2, padding="same", pool_size=[2, 2]
    )(c2)

    flat = layers.Flatten()(mp2)
    dense1 = layers.Dense(units=1024, activation=activations.relu)(flat)
    output = layers.Dense(units=num_classes, activation=activations.softmax)(dense1)

    cnn_model = models.Model(inputs=input, outputs=output)
    cnn_model.compile(
        optimizer = optimizers.Adagrad(),
        loss=losses.categorical_crossentropy,
        metrics=[metrics.categorical_accuracy,metrics.binary_accuracy,metrics.mean_squared_error]
    )

    return cnn_model

def main(args):
    model = get_cnn_model()
    model.summary()
    cnn_est = tf.keras.estimator.model_to_estimator(model, model_dir="./models/")

    # input functions

    train_ip_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": x_train},
        y=y_train,
        batch_size=batch_size,
        num_epochs=None,
        shuffle=True
    )

    eval_ip_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": x_test},
        y=y_test,
        num_epochs=1,
        shuffle=True
    )


    # Configure train operation
    train_configuration = tf.estimator.RunConfig()
    train_configuration.model_dir = "./model/"
    train_configuration.save_summary_steps = 50
    train_configuration.save_checkpoints_steps = 100

    cnn_est.config = train_configuration

    cnn_est.train(
        input_fn=train_ip_fn,
        steps=2,
    )
    eval_result = cnn_est.evaluate(
        input_fn=eval_ip_fn, steps=1
    )

    print(eval_result)

    return

if __name__ == '__main__':
    tf.app.run(main)
