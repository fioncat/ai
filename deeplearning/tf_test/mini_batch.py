#!/usr/bin/python
# coding=utf-8


def batches(batch_size, features, labels):
    """
    Create batches of features and labels

    :param batch_size: the batch size(max)
    :param features: list of features.
    :param labels: list of labels
    :return: Batches of (features, labels)
    """
    assert len(features) == len(labels)

    output = []

    for start in range(0, len(features), batch_size):
        end = start + batch_size
        batch = [features[start:end], labels[start:end]]
        output.append(batch)

    return output


if __name__ == '__main__':

    from tensorflow.examples.tutorials.mnist import input_data
    import tensorflow as tf
    import numpy as np

    learning_rate = 0.001
    n_input = 784
    n_classes = 10

    # origin mnist data
    mnist = input_data.read_data_sets('/datasets/ud730/mnist', one_hot=True)

    # features data
    train_features = mnist.train.images
    test_features = mnist.test.images

    # labels data
    train_labels = mnist.train.labels.astype(np.float32)
    test_labels = mnist.test.labels.astype(np.float32)

    # features and labels Tensor
    # The first dimension is None to save the batch size.
    features = tf.placeholder(tf.float32, [None, n_input])
    labels = tf.placeholder(tf.float32, [None, n_classes])

    # Initialize weights and bias
    weights = tf.Variable(tf.random_normal([n_input, n_classes]))
    bias = tf.Variable(tf.random_normal([n_classes]))

    # Linear Model
    logits = tf.add(tf.matmul(features, weights), bias)

    # Define loss and optimizer.
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

    # Calculate accuracy
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    batch_size = 128

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        for batch_features, batch_label in batches(batch_size, train_features, train_labels):
            sess.run(optimizer, feed_dict={features: batch_features, labels: batch_label})

        test_accuracy = sess.run(accuracy, feed_dict={features: test_features, labels: test_labels})

    print("test accuracy: {}".format(test_accuracy))
