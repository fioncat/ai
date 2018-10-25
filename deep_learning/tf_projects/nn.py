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
        self.train_op = None
        self.tf_pred = None
        self.tf_accuracy = None

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
        """Define Output Layer

        The output layer receives the input of the hidden layer and does the final step.
        It is necessary to specify the type of the output layer and the number of labels
        to be output.
        Generally, how many nodes are there in the label output layer.
        If you need to output a single probability, use the Sigmoid output.
        If you need to output multiple probabilities, use Softmax output.
        Using logits will get the most raw output.

        :param n_labels: Number of labels, equals to number of nodes in output layer.
        :param output_layer_type: You have 3 choices:
                1. Sigmoid: Output a single probability,
                   note that if you use Sigmoid then nLabels must be 1 (single-class task)
                2. Softmax: The most commonly used option, output multiple probabilities
                   (constituting a discrete probability distribution) for multi-classification tasks.
                3. logits: Directly use the output of the hidden layer,
                   which will output the score of each unit instead of the probability.
        :param output_type: dtype for output, default is tf.float32
        :param output_name: output tensor name
        """

        self.n_labels = n_labels
        self.output_layer_type = output_layer_type
        self.output_type = output_type
        self.output_name = output_name

    def train_process(self, learning_rate=0.01, reg=None, reg_lambda=0.0001,
                      optimizer_type='adam', loss_type='cross_entropy',
                      avg_decy=0.99):
        """Define Train Process

        Define some behaviors and hyperparameters during the training process.
        These properties have a great impact on training and it is recommended
        to debug and modify them carefully.

        :param learning_rate: Global Learning Rate.
        :param reg: Regularization, have three choices:
                1. 'l1': use L1 Regularization.
                2. 'l2': use L2 Regularization.
                3. None: do no use Regularization.(default)

        :param reg_lambda: If Regularization is not used, this parameter is useless.
                else, this parameter control the degree of regularization
        :param optimizer_type: The type of Optimizer, have three choices:
                1. 'adam': use AdamOptimizer.(default, recommended)
                2. 'gradient_descent': use GradientDescentOptimizer.

        :param loss_type: The loss function type, have two choices:
                1. 'cross_entropy': use cross entropy(default)
                2. ';

        :param avg_decy: Average Decy
        """
        self.learning_rate = learning_rate
        self.reg = reg
        self.reg_lambda = reg_lambda
        self.optimizer_type = optimizer_type
        self.loss_type = loss_type
        self.avg_decay = avg_decy

    def compile(self):
        """Compile The model

        Make sure you have called the three functions input_layer(),
        hidden_layer(), output_layer() and train_process() before calling this method.
        Otherwise, the compilation process will fail.

        Note that we won't check that you have called these three functions.
        If you compile without calling these three functions, an exception will be thrown.

        After compiling, it is recommended that you call summary()
        to see if your model meets your expectations.

        """

        # In training, these tensors need to be assigned.
        # In the prediction, only x and kp need to be assigned.
        # The kp in prediction is generally taken as 1...
        self.tf_x = tf.placeholder(dtype=self.input_type, shape=(None, self.n_features))
        self.tf_y = tf.placeholder(dtype=self.output_type, shape=(None, self.n_labels))
        self.tf_kp = tf.placeholder(dtype=tf.float32)

        # Get the number of nodes in each layer, including the input and output layers.
        # This information will be used for summary output.
        # Is also needed when constructing parameters.
        if type(self.hidden_shape) == list:   # Multi-hidden layer
            self.layers = self.hidden_shape.copy()
            self.layers.insert(0, self.n_features)
            self.layers.append(self.n_labels)
        elif type(self.hidden_shape) == int:  # Single hidden layer
            self.layers = [self.n_features, self.hidden_shape, self.n_labels]
        else:
            raise TypeError('hidden_shape must be int or list, found',
                            str(type(self.hidden_shape)))
        self.n_layers = len(self.layers)

        ##########################################################
        # Forward Propagation, Get the output of hidden layer(s).#
        ##########################################################
        self.tf_logits = self.inference(None)

        # Convert the logits output to the final output
        if self.output_layer_type == 'sigmoid':
            self.tf_output = tf.nn.sigmoid(self.tf_logits)
        elif self.output_layer_type == 'softmax':
            self.tf_output = tf.nn.softmax(self.tf_logits)
        elif self.output_layer_type == 'logits':
            self.tf_output = self.tf_logits
        else:
            raise TypeError('Unknown output_layer_type', self.output_layer_type)

        # Applied moving average.
        tf_global_epoch = tf.Variable(0, trainable=False)
        tf_var_avg = tf.train.ExponentialMovingAverage(self.avg_decay, tf_global_epoch)
        tf_var_avg_op = tf_var_avg.apply(tf.trainable_variables())

        # TODO: avg logits desc
        self.tf_avg_logits = self.inference(tf_var_avg)

        # Now the neural network processing input and gets the logits output
        if self.loss_type == 'cross_entropy':
            tf_loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.tf_y,
                                                              logits=self.tf_logits)
            tf_loss = tf.reduce_mean(tf_loss)
        else:
            raise ValueError('Unknown loss_type', self.loss_type)

        # If regularization is required, the loss is added after the already
        # calculated regularization term, resulting in the final loss.
        if self.reg is not None:
            tf.add_to_collection('loss', tf_loss)
            tf_loss = tf.add_n(tf.get_collection('loss'))

        # Define optimizer.
        if self.optimizer_type == 'adam':
            tf_optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(tf_loss)
        elif self.optimizer_type == 'gradient_descent':
            tf_optimizer = tf.train.GradientDescentOptimizer(
                self.learning_rate).minimize(tf_loss)
        else:
            raise TypeError('Unknown optimizer_type', self.optimizer_type)

        # TODO: control dependencies desc
        with tf.control_dependencies([tf_optimizer, tf_var_avg_op]):
            self.train_op = tf.no_op(name='train')

        # Calculating the correct rate is mainly to observe the correct rate
        # of the verification set during training to perform operations such as early termination.
        self.tf_pred = tf.equal(tf.argmax(self.tf_avg_logits, 1), tf.argmax(self.tf_y, 1))
        self.tf_accuracy = tf.reduce_mean(tf.cast(self.tf_pred, tf.float32))

    def summary(self):
        """
        Print out the information of the layer of the model.
        It includes the number of nodes and parameters of each layer.
        By this method, you can understand whether the model meets your expectations.
        It is recommended to call this method to check your model before training.
        """

        desc = ['>>>>>>>>>>>>>> Model Overview <<<<<<<<<<<<<<',
                'nFeatures: {}'.format(self.n_features),
                'nLabels: {}'.format(self.n_labels),
                'Network structure:']
        total_w = 0
        total_b = 0
        total = 0
        for i in range(len(self.layers)):
            desc.append('============================================')
            desc.append('Layer - {}/{}'.format((i + 1), self.n_layers))
            desc.append('--------------------------------------------')
            if i == 0:
                desc.append('Layer Type: Input Layer')
            elif i == self.n_layers - 1:
                desc.append('Layer Type: Output Layer')
            else:
                desc.append('Layer Type: Hidden Layer')

            desc.append('nNodes: {}'.format(self.layers[i]))
            if i != 0:
                n_weights = self.layers[i - 1] * self.layers[i]
                n_biases = self.layers[i]
                desc.append('nParameters:')
                desc.append('\tWeights:{}'.format(n_weights))
                desc.append('\tBiases:{}'.format(n_biases))
                desc.append('\tTotal:{}'.format(n_weights + n_biases))

                total_w += n_weights
                total_b += n_biases
                total += (n_weights + n_biases)
            if i == self.n_layers - 1:
                desc.append('Output Type:{}'.format(self.output_layer_type))

        desc.append('============================================')
        desc.append('Total nParameters:')
        desc.append('\tWeights:{}'.format(total_w))
        desc.append('\tBiases:{}'.format(total_b))
        desc.append('\tTotal:{}'.format(total))
        desc.append('Training information:')
        desc.append('\tLearning Rate:{}'.format(self.learning_rate))
        desc.append('\tLoss Type:{}'.format(self.loss_type))
        desc.append('\tUse regularization:{}'.format(self.reg))
        desc.append('\tOptimizer Type:{}'.format(self.optimizer_type))

        print('\n'.join(desc))

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
    model.hidden_layer(128)
    model.output_layer(10)
    model.train_process()

    model.compile()

    model.summary()
