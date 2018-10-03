import tensorflow as tf

# 创建TensorFlow对象
hello_constant = tf.constant('Hello TensorFlow!')

with tf.Session() as sess:
    output = sess.run(hello_constant)
    print(output)
