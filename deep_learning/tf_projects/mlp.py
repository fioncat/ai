#!/usr/bin/python

# Multilayer Perceptron based on Tensorflow.
# You can define and train the deep neural network with
# some simple calls.

import tensorflow as tf
import numpy as np


class DeepFFNN(object):
    """

    """

    def __init__(self):
        """

        """

        self.build_ok = False
        self.last_train = None
        self.structure = []

        self.x = None
        self.y = None
        self.lr = None

        self.loss = None
        self.optimizer = None
        self.output = None

    def build(self, input_shape, hidden_shape, output_shape,
              loss_type='softmax_cross_entropy',
              reg=None, _lambda=0.1,
              optimizer_type='adam',
              input_type=tf.float32, output_type=tf.float32,
              weights_init=None, bias_init=0.1,
              output_layer_type='logits', output_name='output'):
        """

        :param input_shape:
        :param hidden_shape:
        :param output_shape:
        :param loss_type:
        :param reg:
        :param _lambda:
        :param optimizer_type:
        :param input_type:
        :param output_type:
        :param weights_init:
        :param bias_init:
        :param output_layer_type:
        :param output_name:
        :return:
        """

        # network input, learning rate
        self.x = tf.placeholder(dtype=input_type, shape=(None, input_shape))
        self.y = tf.placeholder(dtype=output_type, shape=(None, output_shape))
        self.lr = tf.placeholder(dtype=tf.float32)

        # hidden_shape dose not include input and output layers, append them
        # to the list. (head and tail)
        hidden_shape.insert(0, input_shape)
        hidden_shape.append(output_shape)
        self.structure = hidden_shape.copy()
        n_layers = len(hidden_shape)

        # Build input layer.
        cur_layer = self.x
        in_d = hidden_shape[0]

        # Build multi-hidden-layers.
        for i in range(1, n_layers):
            out_d = hidden_shape[i]

            weights = self.get_weights([in_d, out_d], weights_init, reg, _lambda)
            biases = tf.Variable(tf.constant(bias_init, shape=[out_d]))

            cur_layer = tf.nn.relu(tf.add(tf.matmul(cur_layer, weights), biases))

            in_d = hidden_shape[i]

        # Build output layer.
        self.output = self.get_output(cur_layer, output_layer_type)
        tf.identity(self.output, output_name)

        # Loss and Optimizer for training
        self.loss = self.get_loss(cur_layer, self.y, loss_type, reg)

        self.optimizer = self.get_optimizer(self.loss, self.lr, optimizer_type)

    def summary(self):
        pass

    def train(self):
        pass

    def predict(self):
        pass

    def save_model(self):
        pass

    def load_model(self):
        pass

    def check_build(self):
        if not self.build_ok:
            raise RuntimeError("You have not built the network yet," +
                               "please call build() method first.")

    @staticmethod
    def get_weights(shape, init, reg, _lambda):

        if init is None:
            use_default = True
            init_type = 'normal'
        else:
            use_default = len(init) == 1
            name = init[0]
            if name == 'normal':
                init_type = 'normal'
            else:
                init_type = ''

        if init_type == 'normal':
            if use_default:
                var = tf.random_normal(shape)
            else:
                var = tf.random_normal(
                    shape, stddev=init[2], mean=init[1])
        else:
            if use_default:
                var = tf.random_uniform(shape)
            else:
                var = tf.random_uniform(shape, minval=init[0],
                                        maxval=init[1])

        if reg == 'l1':
            tf.add_to_collection('cost', tf.contrib.layers.
                                 l1_regularizer(_lambda)(var))
        elif reg == 'l2':
            tf.add_to_collection('cost', tf.contrib.layers.
                                 l2_regularizer(_lambda)(var))

        return var

    @staticmethod
    def get_output(hidden_output, output_type):
        if output_type == 'sigmoid':
            return tf.nn.sigmoid(hidden_output)
        elif output_type == 'softmax':
            return tf.nn.softmax(hidden_output)
        elif output_type == 'logits':
            return hidden_output
        else:
            DeepFFNN.raise_unknown('output_type', output_type,
                                   ['sigmoid', 'softmax', 'logits'])

    @staticmethod
    def get_loss(_y, y, loss_type, reg):
        loss = None
        if loss_type == 'softmax_cross_entropy':
            loss = tf.nn.softmax_cross_entropy_with_logits(logits=_y, labels=y)
        elif loss_type == 'sigmoid_cross_entropy':
            loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=_y, labels=y)
        elif loss_type == 'square':
            loss = tf.square(_y - y)
        else:
            DeepFFNN.raise_unknown('loss_type', loss_type,
                                   ['softmax_cross_entropy',
                                    'sigmoid_cross_entropy', 'square'])
        loss = tf.reduce_mean(loss)
        if reg is not None:
            tf.add_to_collection('cost', loss)
            return tf.add_n(tf.get_collection('cost'))
        else:
            return loss

    @staticmethod
    def get_optimizer(loss, lr, optimizer_type):
        optimizer = None
        if optimizer_type == 'adam':
            optimizer = tf.train.AdamOptimizer(lr).minimize(loss)
        elif optimizer_type == 'adadelta':
            optimizer = tf.train.AdadeltaOptimizer(lr).minimize(loss)
        elif optimizer_type == 'gradient_descent':
            optimizer = tf.train.GradientDescentOptimizer(lr).minimize(loss)
        else:
            DeepFFNN.raise_unknown('optimizer_type', optimizer_type,
                                   ['adam', 'adadelta', 'gradient_descent'])
        return optimizer

    @staticmethod
    def raise_unknown(wrong_id, wrong, rights):
        error_info = 'Unknown %s "%s", please choose one from %s'\
                        % (wrong_id, wrong, ','.join(rights))
        raise ValueError(error_info)
