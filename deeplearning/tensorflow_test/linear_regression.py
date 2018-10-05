#!/usr/bin/python
# coding=utf-8

# 使用TensorFlow实现线性回归
# 并对mnist数据进行预测

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data


def initialize_normal_weights(n_features, n_labels):
    """
    Initialize Tensorflow Linear Regression weights

    Args:
        n_features: Number of features.
        n_labels: Number of labels.
    
    Returns:
        Tensorflow weights conforms to Normal Distribution.
    """
    return tf.Variable(tf.truncated_normal((n_features, n_labels)))


def initialize_biases(n_labels):
    """
    Initialize Tensorflow Bias.(all zeros)

    Returns:
        Biases, they are all zeros.
    """
    return tf.Variable(tf.zeros(n_labels))


def linear(input, w, b):
    """
    Return linear function in Tensorflow

    Args:
        input: x, Tensorflow Variable.
        w: weights, Tensorflow Variable.
        b: bias, Tensorflow Variable.

    Returns:
        Tensorflow Linear function
    """
    # return (input*w + b)
    return tf.add(tf.matmul(input, w), b)


def mnist_feature_labels(n_labels):
    """
    Get the first n labels from the MNIST dataset.

    Args:
        n_labels: Number of labels to use.

    Returns:
        Tuple of feature list and label list.
    """
    mnist_features = []
    mnist_labels = []

    mnist = input_data.read_data_sets('/datasets/ud730/mnist', one_hot=True)

    for mnist_feature, mnist_label in zip(*mnist.train.next_batch(10000)):

        if mnist_feature[:n_labels].any():
            mnist_features.append(mnist_feature)
            mnist_labels.append(mnist_label[:n_labels])

    return mnist_features, mnist_labels


if __name__ == '__main__':
    n_features = 784
    n_labels = 3

    features = tf.placeholder(tf.float32)
    labels = tf.placeholder(tf.float32)

    w = initialize_normal_weights(n_features, n_labels)
    b = initialize_biases(n_labels)

    logits = linear(features, w, b)

    train_features, train_labels = mnist_feature_labels(n_labels)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        prediction = tf.nn.softmax(logits)

        cross_entropy = tf.reduce_sum(labels * tf.log(prediction), reduction_indices=1)

        loss = tf.reduce_mean(cross_entropy)

        learning_rate = 0.08

        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

        _, l = sess.run([optimizer, loss],
                        feed_dict={features: train_features, labels: train_labels})

print('loss: {}'.format(l))
