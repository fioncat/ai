#!/usr/bin/python
import tensorflow as tf

g1 = tf.Graph()
with g1.as_default():
    v = tf.

g2 = tf.Graph()
with g2.as_default():
    v = tf.constant(2)

with tf.Session(graph=g1) as sess:
    tf.global_variables_initializer().run()
    print(sess.run(v))
