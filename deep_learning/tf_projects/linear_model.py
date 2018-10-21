#!/usr/bin/python
# coding=utf-8

# TensorFlow 实现简单的线性模型
import tensorflow as tf
from numpy.random import RandomState

# 模型输入
x = tf.placeholder(tf.float32, shape=(None, 2), name='x')
y = tf.placeholder(tf.float32, shape=(None, 1), name='y')

# 模型参数
w1 = tf.Variable(tf.random_normal([2, 3], seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], seed=1))

# 模型计算
hidden = tf.matmul(x, w1)
hidden = tf.matmul(hidden, w2)

# 模型输出
output = tf.sigmoid(hidden)

# 定义损失函数
cost = -tf.reduce_mean(y * tf.log(tf.clip_by_value(output, 1e-10, 1.0)) +
                       (1 - y) * tf.log(tf.clip_by_value(1 - output, 1e-10, 1.0)))

# 定义优化器
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

# 随机创建一个数据集
rdm = RandomState(1)
data_size = 128
X = rdm.rand(data_size, 2)    # 创建128个坐标点
Y = [[int(x1 + x2 < 1)] for (x1, x2) in X]  # 创建标签

# 一些超参数
batch_size = 8          # 每次训练的数据大小
epochs = 5000           # 迭代轮数
print_per_epoch = 500   # 每隔多少轮打印一次训练情况

with tf.Session() as sess:

    # 初始化变量
    sess.run(tf.global_variables_initializer())

    # 开始训练
    for epoch in range(epochs):

        # 每次训练的数据分段
        start = (epoch * batch_size) % data_size
        end = min(start + batch_size, data_size)

        # 执行优化器,训练参数
        sess.run(optimizer, feed_dict={
            x: X[start:end], y: Y[start:end]
        })

        # 每隔一段时间,打印训练情况
        if (epoch + 1) % print_per_epoch == 0:
            # 计算当前代价
            loss = sess.run(cost, feed_dict={
                x: X[start:end], y: Y[start:end]
            })

            # 打印代价
            print("Epoch {}/{}: Loss:{:.4f}".format(epoch + 1, epochs, loss))

