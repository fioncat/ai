#!/usr/bin/python
# coding=utf-8

# 使用Keras训练CIFAR-10数据

from keras.datasets import cifar10
from keras.utils import np_utils, to_categorical

import numpy as np

if __name__ == '__main__':

    # 数据集可以直接获取
    # 如果下载失败,可以手动下载并放到家目录的.keras/datasets中
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # 数据预处理,将每个像素点的范围从[0,255]转换为[0,1]
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    # 对label进行one-hot编码
    num_classes = len(np.unique(y_train))
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    # 从训练集中抽取验证集
    x_train, x_valid = x_train[5000:], x_train[:5000]
    y_train, y_valid = y_train[5000:], y_train[:5000]

    # 定义CNN结构
    from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout
    from keras.models import Sequential

    model = Sequential()
    model.add(Conv2D(filters=16, kernel_size=2, padding='same', activation='relu',
                     input_shape=(32, 32, 3)))
    model.add(MaxPool2D(pool_size=2))
    model.add(Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=2))
    model.add(Conv2D(filters=64, kernel_size=2, padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=2))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(500, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(10, activation='softmax'))
    model.summary()

    # 编译,训练模型

