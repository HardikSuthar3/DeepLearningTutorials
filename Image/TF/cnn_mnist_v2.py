from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

def cnn_model_fn(features, labels, mode):
    """Model Function for CNN."""
    # Input Layer
    input_layer = tf.reshape(features['x'], shape=[-1, 28, 28, 1])

    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu,
        name="Convolution-1"
    )

    #adding summary
    tf.summary.scalar("conv1/weights",conv1)

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=[2, 2])

    # Convolutional Layer #2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu,
        name="Convolution-2"
    )

    # adding summary
    tf.summary.scalar("conv1/weights", conv2)

    # Pooling Layer #2
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=[2, 2])

    # Dense Layer
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
    dense = tf.layers.dense(
        inputs=pool2_flat,
        units=1024,
        activation=tf.nn.relu,
        name="Dense"
    )
    dropout = tf.layers.dropout(
        inputs=dense,
        rate=0.4,
        training=(mode == tf.estimator.ModeKeys.TRAIN),
        name="Dropout_Layer"
    )
    # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=10)

    predictions = {
        "classes": tf.argmax(input=logits, axis=1, name="Class_Prediction"),
        "probabilities": tf.nn.softmax(logits=logits, name="Softmax")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if (mode == tf.estimator.ModeKeys.TRAIN):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])
    }
    return tf.estimator.EstimatorSpec(mode=mode, eval_metric_ops=eval_metric_ops, loss=loss)

def main(argv):
    # Load training and eval data
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_data = mnist.train.images  # Returns np.array
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    eval_data = mnist.test.images  # Returns np.array
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    # Create the Estimator
    my_checkpoint_config = tf.estimator.RunConfig(save_checkpoints_steps=100)
    classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn,
        model_dir="./model/",
        config=my_checkpoint_config
    )

    tensor_to_log = {"Probabilities": "Softmax"}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensor_to_log, every_n_iter=50)

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=100,
        num_epochs=None,
        shuffle=True)

    # Train model
    classifier.train(input_fn=train_input_fn, hooks=[logging_hook], steps=20000)

    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)
    eval_result = classifier.evaluate(input_fn=eval_input_fn)

    print(eval_result)

    return

# Our application logic will be added here
if __name__ == "__main__":
    tf.app.run()
