#!/usr/bin/python

import tensorflow as tf


class DeepNeuralNetwork(object):
    """

    """

    def __init__(self):
        """
        Constructor for DeepNeuralNetwork.
        All attributes will be set by user.
        """

        # Input layer
        self.n_features = None
        self.input_name = None
        self.input_type = None

        # hidden layer
        self.weights_init = None
        self.bias_init = None
        self.hidden_shape = None

        self.layers = None
        self.n_layers = None

        # Moving average
        self.avg_decay = None

        # Output layer
        self.n_labels = None
        self.output_type = None
        self.output_name = None
        self.output_layer_type = None

        # Loss
        self.loss_type = None

        # Regularization
        self.reg = None
        self.reg_lambda = None

        # Optimizer
        self.optimizer_type = None

        # Learning rate
        self.learning_rate = None

        self.all_weights = []
        self.all_biases = []

        # Model Tensor(Compilation Target)
        self.tf_x = None
        self.tf_y = None
        self.tf_kp = None
        self.tf_logits = None
        self.tf_avg_logits = None
        self.tf_output = None
        self.tf_loss = None
        self.tf_optimizer = None
        self.tf_accuracy = None
        self.tf_global_epoch = None
        self.train_op = None
        self.tf_pred = None

    def input_layer(self, n_features, input_name='input', input_type=tf.float32):
        """Define input layer.

        DeepNeuralNetwork only accept one-dimension inputs.
        That means you need to parse #features to the model.
        If your input is a matrix, please flatten it first.

        :param n_features: The number of features.
        :param input_name: Input Tensor's name.
        :param input_type: The type of input data, tensorflow type,
        default is tf.float32.
        """

        self.n_features = n_features
        self.input_name = input_name
        self.input_type = input_type

    def hidden_layer(self, hidden_shape, weights_init=None, bias_init=0.1):
        """Define hidden layers.

        Hidden Layers are all fully-connected layers(Dense Connect).

        :param hidden_shape: If it is a integer, there will only be one hidden
        layer, and this parameter indicates the number of nodes in this hidden layer.
        If it is an array, there will be more than one hidden layer,
        the number of the layers equals to the length of the array.
        The nth element of the array indicates the number of
        nodes in the nth hidden layer.

        For example, [256, 128, 64] indicates there are three hidden layers, the
        numbers of nodes is 256, 128, 64.
        512 indicates there is only one hidden layer with 512 nodes.

        :param weights_init:This parameter determines how the weights in the hidden
        layer are initialized. By default, the Gaussian distribution is used to
        initialize the weights. If you want to specify the parameters of the
        distribution and distribution, set this parameter to an array. The first
        element of the array. Is a string representing the type of distribution.
        Currently supported distributions and parameters are:
            1. 'normal': Gaussian distribution
                weights_init[1]: mean
                weights_init[2]: stddev
            2. 'uniform': Evenly distributed
                weights_init[1]: min val
                weights_init[2]: max val
        weights_init[1] and weights_init[2] are not required,
        if you only specify weights_init[0], then the default parameters will be used.

        For example, ['normal', 10, 0.5] indicates to use a Gaussian distribution
        with mean=10, stddev=0.5 to initialize weights.
        ['uniform'] indicates to use a Evenly distribution with default parameter to
        initialize the weights.

        :param bias_init:Unlike weights, the initialization of the biases requires
        only one number, and all biases are initialized to this number.
        It is generally recommended to set a small number, the default is 0.1.
        """
        self.hidden_shape = hidden_shape
        self.weights_init = weights_init
        self.bias_init = bias_init

    def output_layer(self, n_labels, output_layer_type='logits',
                     output_type=tf.float32, output_name='output'):

        self.n_labels = n_labels
        self.output_layer_type = output_layer_type
        self.output_type = output_type
        self.output_name = output_name

    def train_process(self, learning_rate=0.01, reg=None, reg_lambda=0.0001,
                      optimizer_type='adam', loss_type='cross_entropy',
                      avg_decy=0.99):
        self.learning_rate = learning_rate
        self.reg = reg
        self.reg_lambda = reg_lambda
        self.optimizer_type = optimizer_type
        self.loss_type = loss_type
        self.avg_decay = avg_decy

    def compile(self):

        self.tf_x = tf.placeholder(dtype=self.input_type, shape=(None, self.n_features))
        self.tf_y = tf.placeholder(dtype=self.output_type, shape=(None, self.n_labels))
        self.tf_kp = tf.placeholder(dtype=tf.float32)

        if type(self.hidden_shape) == list:
            self.layers = self.hidden_shape.copy()
            self.layers.insert(0, self.n_features)
            self.layers.append(self.n_labels)
        elif type(self.hidden_shape) == int:
            self.layers = [self.n_features, self.hidden_shape, self.n_labels]
        else:
            raise TypeError('hidden_shape must be int or list, found',
                            str(type(self.hidden_shape)))

        self.n_layers = len(self.layers)

        self.tf_logits = self.inference(None)

        if self.output_layer_type == 'sigmoid':
            self.tf_output = tf.nn.sigmoid(self.tf_logits)
        elif self.output_layer_type == 'softmax':
            self.tf_output = tf.nn.softmax(self.tf_logits)
        elif self.output_layer_type == 'logits':
            self.tf_output = self.tf_logits
        else:
            raise TypeError('Unknown output_layer_type', self.output_layer_type)

        self.tf_global_epoch = tf.Variable(0, trainable=False)
        tf_var_avg = tf.train.ExponentialMovingAverage(self.avg_decay, self.tf_global_epoch)
        tf_var_avg_op = tf_var_avg.apply(tf.trainable_variables())

        self.tf_avg_logits = self.inference(tf_var_avg)

        if self.loss_type == 'cross_entropy':
            self.tf_loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.tf_y,
                                                                   logits=self.tf_logits)
            self.tf_loss = tf.reduce_mean(self.tf_loss)

        if self.reg is not None:
            tf.add_to_collection('loss', self.tf_loss)
            self.tf_loss = tf.add_n(tf.get_collection('loss'))

        if self.optimizer_type == 'adam':
            self.tf_optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.tf_loss)
        elif self.optimizer_type == 'gradient_descent':
            self.tf_optimizer = tf.train.GradientDescentOptimizer(
                self.learning_rate).minimize(self.tf_loss)
        else:
            raise TypeError('Unknown optimizer_type', self.optimizer_type)

        with tf.control_dependencies([self.tf_optimizer, tf_var_avg_op]):
            self.train_op = tf.no_op(name='train')

        self.tf_pred = tf.equal(tf.argmax(self.tf_avg_logits, 1), tf.argmax(self.tf_y, 1))

        self.tf_accuracy = tf.reduce_mean(tf.cast(self.tf_pred, tf.float32))

    def inference(self, avg_class):
        in_d = self.layers[0]
        cur_layer = self.tf_x
        for i in range(1, self.n_layers):
            out_d = self.layers[i]

            if avg_class is None:

                weights = self.init_weights([in_d, out_d], self.weights_init,
                                            self.reg, self.reg_lambda)
                biases = tf.Variable(tf.constant(self.bias_init, shape=[out_d]),
                                     dtype=tf.float32)
                self.all_weights.append(weights)
                self.all_biases.append(biases)

            else:
                weights = avg_class.average(self.all_weights[i - 1])
                biases = avg_class.average(self.all_biases[i - 1])

            cur_layer = tf.add(tf.matmul(cur_layer, weights), biases)
            if i != self.n_layers - 1:
                cur_layer = tf.nn.relu(cur_layer)
                cur_layer = tf.nn.dropout(cur_layer, self.tf_kp)
            in_d = self.layers[i]

        return cur_layer

    def mnist_train(self):
        from tensorflow.examples.tutorials.mnist import input_data
        mnist = input_data.read_data_sets('data/mnist', one_hot=True)
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            valid_feed = {self.tf_x: mnist.validation.images,
                          self.tf_y: mnist.validation.labels,
                          self.tf_kp: 1.}

            for i in range(8000):
                x_batch, y_batch = mnist.train.next_batch(100)
                sess.run(self.train_op, feed_dict={self.tf_x: x_batch,
                                                   self.tf_y: y_batch,
                                                   self.tf_kp: 0.7})
                if i % 1000 == 0:
                    valid_acc = sess.run(self.tf_accuracy, feed_dict=valid_feed)
                    print("valid acc:", valid_acc)

    @staticmethod
    def init_weights(shape, weights_init, reg, reg_lambda):
        if weights_init is None:
            use_default = True
            dis_type = 'normal'
        else:
            dis_type = weights_init[0]
            if len(weights_init) == 1:
                use_default = True
            else:
                use_default = False

        if dis_type == 'normal':
            if use_default:
                dis = tf.random_normal(shape=shape)
            else:
                dis = tf.random_normal(shape=shape,
                                       mean=weights_init[1], stddev=weights_init[2])
        elif dis_type == 'uniform':
            if use_default:
                dis = tf.random_uniform(shape=shape)
            else:
                dis = tf.random_uniform(shape=shape,
                                        minval=weights_init[1], maxval=weights_init[2])
        else:
            raise ValueError('unknown distribution %s' % dis_type)

        var = tf.Variable(dis, dtype=tf.float32)
        if reg == 'l1':
            tf.add_to_collection('loss', tf.contrib.layers.l1_regularizer(reg_lambda)(var))
        elif reg == 'l2':
            tf.add_to_collection('loss', tf.contrib.layers.l2_regularizer(reg_lambda)(var))

        return var


if __name__ == '__main__':

    model = DeepNeuralNetwork()
    model.input_layer(n_features=784)
    model.hidden_layer([128, 64])
    model.output_layer(10)
    model.train_process()

    model.compile()

    model.mnist_train()
