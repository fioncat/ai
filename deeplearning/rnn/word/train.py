#!/usr/bin/python
# coding=utf-8

from load_text import encode_text
from network import CharNetwork

encoded, vocab, _ = encode_text('test.txt')

# Hyper parameters
batch_size = 100        # Sequences per batch
num_steps = 100         # Number of sequence steps per batch
lstm_size = 512         # Size of hidden layers in LSTMs
num_layers = 2          # Number of LSTM layers
learning_rate = 0.001   # Learning rate
keep_prob = 0.5         # Dropout keep probability


model = CharNetwork(len(vocab), batch_size=batch_size, num_steps=num_steps,
                    lstm_size=lstm_size, num_layers=num_layers,
                    learning_rate=learning_rate)

model.train(encoded, keep_prob=keep_prob, epochs=40)
