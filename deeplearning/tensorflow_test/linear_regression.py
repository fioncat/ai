#!/usr/bin/python
#coding=utf-8

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
