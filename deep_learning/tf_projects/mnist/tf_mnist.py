#!/usr/bin/python
# coding=utf-8

# Tensorflow 实现CNN
import tensorflow as tf
import numpy as np
from tensorflow.python.framework import graph_util

# 读取MNIST数据集
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('.', one_hot=True, reshape=False)

# 参数
learning_rate = 0.00001
epochs = 5
batch_size = 128

# 用于验证的样本数
valid_size = 256
test_size = 256

# 10个类别
n_classes = 10

# 保留单元的概率
keep_prob = 0.70

# CNN参数
weights = {
    'conv1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
    'conv2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    'dense': tf.Variable(tf.random_normal([7 * 7 * 64, 1024])),
    'out': tf.Variable(tf.random_normal([1024, n_classes]))
}

biases = {
    'conv1': tf.Variable(tf.random_normal([32])),
    'conv2': tf.Variable(tf.random_normal([64])),
    'dense': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}


def conv2d(x, W, b, strides=1):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')


def network(x, weights, biases, keep_prob):

    conv1 = conv2d(x, weights['conv1'], biases['conv1'])
    conv1 = maxpool2d(conv1)

    conv2 = conv2d(conv1, weights['conv2'], biases['conv2'])
    conv2 = maxpool2d(conv2)

    # Flatten First
    dense = tf.reshape(conv2, [-1, weights['dense'].get_shape().as_list()[0]])
    dense = tf.add(tf.matmul(dense, weights['dense']), biases['dense'])
    dense = tf.nn.relu(dense)
    dense = tf.nn.dropout(dense, keep_prob)

    return tf.add(tf.matmul(dense, weights['out']), biases['out'])


def print_imp(last_loss, current_loss):
    if last_loss == 'inf':
        imp = True
    else:
        imp = current_loss < last_loss

    if imp:
        print('Loss improved from ', last_loss, ' to ', current_loss)
    else:
        print('Loss did not improve from ', last_loss, ' to ', current_loss)


# 输入数据
x = tf.placeholder(tf.float32, [None, 28, 28, 1])
y = tf.placeholder(tf.float32, [None, n_classes])
kp = tf.placeholder(tf.float32)

# 网络处理数据后产生的logits输出
logits = network(x, weights, biases, kp)

# 损失和优化器,用于训练网络
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

# 准确度
pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(pred, tf.float32))

# 用于初始化变量
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    last_loss = 'inf'
    for epoch in range(epochs):

        losses = []
        valid_accs = []
        for batch in range(mnist.train.num_examples // batch_size):

            batch_x, batch_y = mnist.train.next_batch(batch_size)
            sess.run(optimizer, feed_dict={
                x: batch_x, y: batch_y, kp: keep_prob
            })

            loss = sess.run(cost, feed_dict={
                x: batch_x, y: batch_y, kp: 1.
            })
            losses.append(loss)

            valid_acc = sess.run(accuracy, feed_dict={
                x: mnist.validation.images[:valid_size],
                y: mnist.validation.labels[:valid_size],
                kp: 1.
            })
            valid_accs.append(valid_acc)

        current_loss = np.mean(np.array(losses))

        print("===================================================")
        print("Epoch - {:>2}".format(epoch))
        print("Validation Accuracy: {:.6f}".format(max(valid_accs)))
        print_imp(last_loss, current_loss)
        last_loss = current_loss
        print("===================================================")

    test_acc = sess.run(accuracy, feed_dict={
        x: mnist.test.images[:test_size],
        y: mnist.test.labels[:test_size],
        kp: 1.
    })
    print()
    print("Train Over, test accuracy: {}".format(test_acc))

    constant_graph = graph_util.convert_variables_to_constants(sess,
                                                               sess.graph_def, ['op_to_store'])
    with tf.gfile.FastGFile('model/mnist.pb', mode='wb') as f:
        f.write(constant_graph.SerializeToString())


