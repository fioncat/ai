if __name__ == '__main__':

    save_file = 'models/mnist/mnist.ckpt'

    import tensorflow as tf
    from tensorflow.examples.tutorials.mnist import input_data

    mnist = input_data.read_data_sets('.', one_hot=True, reshape=False)

    print('MNIST data read done.')

    # Parameters
    learning_rate = 0.001
    training_epochs = 30
    batch_size = 128
    display_step = 1

    # data feature
    n_input = 784
    n_classes = 10

    # Layer number of features
    n_hidden_layer = 256

    # Initialize the network's Parameters
    weights = {
        'hidden_layer': tf.Variable(tf.random_normal([n_input, n_hidden_layer])),
        'out': tf.Variable(tf.random_normal([n_hidden_layer, n_classes]))
    }
    biases = {
        'hidden_layer': tf.Variable(tf.random_normal([n_hidden_layer])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }

    # TF Graph input
    x = tf.placeholder(tf.float32, [None, 28, 28, 1])
    y = tf.placeholder(tf.float32, [None, n_classes])
    x_flat = tf.reshape(x, [-1, n_input])

    # Define two-layer neural network model
    hidden_layer = tf.add(tf.matmul(x_flat, weights['hidden_layer']),
                          biases['hidden_layer'])
    hidden_layer = tf.nn.relu(hidden_layer)
    logits = tf.add(tf.matmul(hidden_layer, weights['out']), biases['out'])

    # Define Optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=y))
    optimizer = tf.train.GradientDescentOptimizer(
        learning_rate=learning_rate).minimize(cost)

    # Define accuracy
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, save_file)

        test_accuracy = sess.run(accuracy, feed_dict={
            x: mnist.test.images,
            y: mnist.test.labels
        })

    print("Test Accuracy: {}".format(test_accuracy))
