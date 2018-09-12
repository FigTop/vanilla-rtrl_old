#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 16:30:58 2018

@author: omarschall
"""

import numpy as np
from network import RNN
from utils import *
from gen_data import gen_data
import matplotlib.pyplot as plt
import time

X, Y = gen_data(40000, 3, 7)

n_in     = 1
n_hidden = 32
n_out    = 1

W_in  = np.random.normal(0, np.sqrt(1/(n_in + n_hidden)), (n_hidden, n_in))
W_rec = np.random.normal(0, np.sqrt(1/(n_hidden + n_hidden)), (n_hidden, n_hidden))
W_out = np.random.normal(0, np.sqrt(1/(n_hidden + n_hidden)), (n_out, n_hidden))

b_rec = np.zeros(n_hidden)
b_out = np.zeros(n_out)

rnn = RNN(W_in, W_rec, W_out, b_rec, b_out, activation=relu, output=sigmoid, loss=sigmoid_cross_entropy)

t1 = time.time()
losses, y_hats = rnn.run(X, Y, learning_rate=0.005, method='rtrl')
t2 = time.time()

print(t2 - t1)

plt.plot(np.convolve(losses, np.ones(100)/100, mode='valid'))
plt.ylim([0,1])
#plt.plot(losses, '.', alpha=0.4)
