#!/usr/bin/python
# coding=utf-8

# 使用Keras训练CIFAR-10数据

import keras
from keras.datasets import cifar10

# 数据集可以直接获取
# 如果下载失败,可以手动下载并放到家目录的.keras/datasets中
(x_train, y_train), (x_test, y_test) = cifar10.load_data()


