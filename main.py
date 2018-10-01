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

X, Y = gen_data(100000, 3, 6, one_hot=True, deterministic=True)

n_in     = 2
n_hidden = 32
n_out    = 2

W_in  = np.random.normal(0, np.sqrt(1/(n_in + n_hidden)), (n_hidden, n_in))
W_rec = np.random.normal(0, np.sqrt(1/(n_hidden + n_hidden)), (n_hidden, n_hidden))
W_out = np.random.normal(0, np.sqrt(1/(n_hidden + n_hidden)), (n_out, n_hidden))

b_rec = np.zeros(n_hidden)
b_out = np.zeros(n_out)

alpha = 1

rnn = RNN(W_in, W_rec, W_out, b_rec, b_out,
          activation=relu,
          alpha=alpha,
          output=softmax,
          loss=softmax_cross_entropy)


#optimizer = Adam(lr=0.001)
optimizer = SGD(lr=0.05)

t1 = time.time()
rnn.run(X, Y, optimizer, method='kf', monitors=['A', 'W_rec', 'grads', 'loss_', 'u', 'a', 'h'])
#rnn.run(X, Y, optimizer, method='rtrl', monitors=['W_rec', 'grads', 'loss_', 'a', 'h'])
t2 = time.time()
print('Time Elapsed:'+str(t2 - t1))

A = np.array(rnn.mons['A'])
W = np.array(rnn.mons['W_rec'])

A_radii = get_spectral_radii(A)
W_radii = get_spectral_radii(W)
u_norms = [np.sum(np.square(u)) for u in rnn.mons['u']]
u_norms /= np.amax(u_norms)
A_T = np.swapaxes(A, 1, 2)
dAdt = A_T[1:,:,:] - A_T[:-1,:,:]
dAdt = A[1:,:,:] - A[:-1,:,:]
dWdt = W[1:,:,:] - W[:-1,:,:]
alignment = [0] + get_vector_alignment(dAdt, dWdt)

signals = [rnn.mons['loss_'], W_radii, A_radii, u_norms, alignment]
fig = plot_filtered_signals(signals, y_lim=[-0.1, 1.5])
plt.legend(['Loss', 'W Spec. Rad.', 'A Spec. Rad.', 'u Norm', 'A-W Update Alignment'])

#A_norms = np.array([np.sum(np.square(a)) for a in rnn.mons['A']])
#A_norms /= np.amax(A_norms)
#A_norms += 0.5





