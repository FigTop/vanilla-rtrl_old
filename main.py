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
from optimizers import *
from analysis_funcs import *

X, Y = gen_data(5000, 3, 6, one_hot=True, deterministic=True)

n_in     = 2
n_hidden = 32
n_out    = 2

W_in  = np.random.normal(0, np.sqrt(1/(n_in)), (n_hidden, n_in))
W_rec = np.random.normal(0, np.sqrt(1/(n_hidden)), (n_hidden, n_hidden))
#W_rec = np.eye(n_hidden)*0.54
W_out = np.random.normal(0, np.sqrt(1/(n_hidden)), (n_out, n_hidden))

A = np.random.normal(0,  1/np.sqrt(n_hidden+n_hidden), (n_hidden, n_hidden))
B = np.random.normal(0, 1/np.sqrt(n_out+n_hidden), (n_hidden, n_out))
C = np.zeros(n_hidden)

b_rec = np.zeros(n_hidden)
b_out = np.zeros(n_out)

alpha = 1

rnn = RNN(W_in, W_rec, W_out, b_rec, b_out,
          activation=relu,
          alpha=alpha,
          output=softmax,
          loss=softmax_cross_entropy,
          A=A, B=B, C=C)


optimizer = SGD(lr=0.001, clipnorm=1)
SG_optimizer = SGD(lr=0.0001, clipnorm=0.5)


#Choose monitors
monitors = ['A', 'W_rec', 'grads', 'loss_', 'a', 'h', 'a_J']

t1 = time.time()
rnn.run(X, Y, optimizer, method='dni', monitors=monitors, SG_optimizer=SG_optimizer, l2_reg=0.001, l2_SG=0.001,
        alpha_SG_target=1, n_SG=5)
t2 = time.time()
print('Time Elapsed:'+str(t2 - t1))

A = np.nan_to_num(np.array(rnn.mons['A']))
W = np.nan_to_num(np.array(rnn.mons['W_rec']))

A_radii = get_spectral_radii(A)
W_radii = get_spectral_radii(W)

signals = [rnn.mons['loss_']]
signals += [W_radii, A_radii]
fig = plot_filtered_signals(signals, y_lim=[-0.1, 1.5], plot_loss_benchmarks=True, filter_size=100)
plt.legend(['loss', 'W Spec. Rad.', 'A Spec. Rad.'])











