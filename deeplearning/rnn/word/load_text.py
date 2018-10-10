#!/usr/bin/python

import numpy as np


def encode_text(text_file):
    """
    Load text file and convert it into integers.

    :param text_file: text file'e path
    :return: encoded ndarray, vocab to index map, index to vocab map
    """

    with open(text_file, 'r') as file:
        text = file.read()

    vocab = sorted(set(text))
    vocab_int = {ch: i for i, ch in enumerate(vocab)}
    int_vocab = dict(enumerate(vocab))

    encoded = np.array([vocab_int[ch] for ch in text], dtype=np.int32)

    return encoded, vocab_int, int_vocab



