#!/usr/bin/python
# coding=utf-8
import numpy as np
import tensorflow as tf
from network import CharNetwork
from load_text import encode_text


def pick_top(preds, vocab_size, top_n=5):
    p = np.squeeze(preds)
    p[np.argsort(p)[:-top_n]] = 0
    p /= np.sum(p)

    c = np.random.choice(vocab_size, 1, p=p)[0]

    return c


def sample(checkpoint, n_samples, lstm_size, prime, text_file):

    global preds, x
    _, vocab_int, int_vocab = encode_text(text_file)

    samples = [c for c in prime]
    model = CharNetwork(len(vocab_int), lstm_size=lstm_size, sampling=True)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, checkpoint)
        new_state = sess.run(model.init_state)
        for ch in prime:
            x = np.zeros((1, 1))
            x[0, 0] = vocab_int[ch]
            preds, mew_state = sess.run([model.prediction, model.final_state],
                                        feed_dict={
                                            model.inputs: x,
                                            model.keep_prob: 1.,
                                            model.init_state: new_state
                                        })
        c = pick_top(preds, len(vocab_int))
        samples.append(int_vocab[c])

        for i in range(n_samples):
            x[0, 0] = c
            preds, new_state = sess.run([model.prediction, model.final_state],
                                        feed_dict={
                                            model.inputs: x,
                                            model.keep_prob: 1.,
                                            model.init_state: new_state
                                        })
            c = pick_top(preds, len(vocab_int))
            samples.append(int_vocab[c])

    return ''.join(samples)


if __name__ == '__main__':
    sample = sample("checkpoints/douluo/char_model.ckpt", 2000, 512, "唐", "data/test_gbk.txt")
    print(sample)
