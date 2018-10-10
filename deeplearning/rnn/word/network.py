#!/usr/bin/python
# coding=utf-8

# Define RNN network
import tensorflow as tf
import numpy as np
from time import time


def input_graph(batch_size, num_steps):
    """
    Define placeholders for inputs, targets, dropout

    :param batch_size: Number of sequences per batch
    :param num_steps: Number of sequence steps in a batch
    :return: inputs, targets, dropout placeholders.
    """
    inputs = tf.placeholder(tf.int32, [batch_size, num_steps], name='inputs')
    targets = tf.placeholder(tf.int32, [batch_size, num_steps], name='targets')

    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    return inputs, targets, keep_prob


def lstm_graph(lstm_size, num_layers, batch_size, keep_prob):
    """
    Define LSTM cell
    :param lstm_size: Size of the hidden layers in the LSTM cell.
    :param num_layers: Number of LSTM layers.
    :param batch_size: Batch size
    :param keep_prob: Scaler tensor for the dropout keep probability.
    """

    def cell_graph(lstm_size, keep_prob):
        # Basic LSTM Cell
        lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
        # Dropout to the cell
        drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)
        return drop

    rnn = tf.contrib.rnn.MultiRNNCell([cell_graph(lstm_size, keep_prob)
                                       for _ in range(num_layers)])
    init_state = rnn.zero_state(batch_size, tf.float32)

    return rnn, init_state


def output_graph(x, in_size, out_size):
    """
    Create a sofrmax layer, return the softmax output and logits.
    :param x: Input tensor
    :param in_size: Size of the input tensor
    :param out_size: Size of the softmax layer
    """
    seq_out = tf.concat(x, axis=1)
    x = tf.reshape(seq_out, [-1, in_size])

    with tf.variable_scope('softmax'):
        softmax_w = tf.Variable(tf.truncated_normal((in_size, out_size), stddev=0.1))
        softmax_b = tf.Variable(tf.zeros(out_size))

    logits = tf.add(tf.matmul(x, softmax_w), softmax_b)

    out = tf.nn.softmax(logits, name='prediction')

    return out, logits


def loss_graph(logits, targets, num_classes):
    """
    Calculate the loss from the logits and the targets

    :param logits: Logits from final fully connected layer
    :param targets: targets for supervised learning
    :param lstm_size: Number of LSTM hidden units
    :param num_classes: Number of classes in targets
    """

    y_one_hot = tf.one_hot(targets, num_classes)
    y = tf.reshape(y_one_hot, logits.get_shape())

    loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y)
    loss = tf.reduce_mean(loss)

    return loss


def optimizer_graph(loss, learning_rate, grab_clip):
    """
    Build optimizer for training, using gradient clipping.
    :param loss: Network loss
    :param learning_rate: Learning rate
    :param grab_clip: gradient clipping
    :return:
    """
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), grab_clip)
    train_op = tf.train.AdadeltaOptimizer(learning_rate)
    optimizer = train_op.apply_gradients(zip(grads, tvars))

    return optimizer


def get_batches(arr, batch_size, n_steps):
    """
    Create a generator that returns batches each call.

    :param arr: Array to make batches.
    :param batch_size: The number of sequences per batch
    :param n_steps: Number of sequence step per batch
    """
    ch_per_batch = batch_size * n_steps
    n_batches = len(arr) // ch_per_batch

    # Keep only enough characters to make full batches.
    arr = arr[:n_batches * ch_per_batch]

    # Reshape into batch size rows.
    arr = arr.reshape((batch_size, -1))

    for i in range(0, arr.shape[1], n_steps):

        # features
        x = arr[:, i:i + n_steps]
        # Targets, shifted by one
        y = np.zeros_like(x)

        y[:, :-1], y[:, -1] = x[:, 1:], x[:, 0]

        yield x, y


class CharNetwork(object):

    def __init__(self, num_classes, batch_size=64, num_steps=50, lstm_size=128,
                 num_layers=2, learning_rate=0.001, grad_clip=5, sampling=False):

        self.batch_size = batch_size
        self.num_steps = num_steps
        self.lstm_size = lstm_size

        # When using network for sampling, will be passing in one character at a time.
        if sampling is True:
            batch_size, num_steps = 1, 1

        tf.reset_default_graph()

        self.inputs, self.targets, self.keep_prob = input_graph(batch_size, num_steps)

        cells, self.init_state = lstm_graph(lstm_size, num_layers, batch_size, self.keep_prob)

        x_one_hot = tf.one_hot(self.inputs, num_classes)

        outputs, state = tf.nn.dynamic_rnn(cells, x_one_hot, initial_state=self.init_state)
        self.final_state = state

        self.prediction, self.logits = output_graph(outputs, lstm_size, num_classes)

        self.loss = loss_graph(self.logits, self.targets, num_classes)
        self.optimizer = optimizer_graph(self.loss, learning_rate, grad_clip)

    def train(self, features, epochs=20, keep_prob=0.75):

        saver = tf.train.Saver(max_to_keep=100)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            counter = 0
            for epoch in range(epochs):

                new_state = sess.run(self.init_state)
                losses = []
                start = time()
                for x, y in get_batches(features, self.batch_size, self.num_steps):
                    counter += 1

                    batch_loss, new_state, _ = sess.run([self.loss, self.final_state, self.optimizer],
                                                        feed_dict={
                                                            self.inputs: x,
                                                            self.targets: y,
                                                            self.keep_prob: keep_prob,
                                                            self.init_state: new_state
                                                        })
                    losses.append(batch_loss)

                end = time()
                print("Epoch :{}/{}     {:.4f} sec".format(epoch + 1, epochs, (end - start)))
                print("Training loss: {:.4f}".format(min(losses)))
                print("============================================")

            saver.save(sess, "checkpoints/douluo/char_model.ckpt")

