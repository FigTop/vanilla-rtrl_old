#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 16:30:58 2018

@author: omarschall
"""

import numpy as np
from network import RNN
from utils import *
from gen_data import *
import matplotlib.pyplot as plt
import time
from optimizers import *
from analysis_funcs import *
from learning_algorithms import *

np.random.seed(2)

#X, Y = coin_task(20000, 3, 6, one_hot=True, deterministic=False)
n_sym = 5
T = 5
X, Y = copy_task(n_sym, 1000, T)

n_in     = n_sym
n_hidden = 32
n_out    = n_sym

W_in  = np.random.normal(0, np.sqrt(1/(n_in)), (n_hidden, n_in))
W_rec = np.random.normal(0, np.sqrt(1/(n_hidden)), (n_hidden, n_hidden))
W_out = np.random.normal(0, np.sqrt(1/(n_out)), (n_out, n_hidden))

b_rec = np.zeros(n_hidden)
b_out = np.zeros(n_out)

alpha = 1

rnn = RNN(W_in, W_rec, W_out, b_rec, b_out,
          activation=relu,
          alpha=alpha,
          output=softmax,
          loss=softmax_cross_entropy)


optimizer = SGD(lr=0.001, clipnorm=1)
SG_optimizer = SGD(lr=0.0001, clipnorm=0.5)
learn_alg = RTRL(rnn)#, 3, 10)#, SG_optimizer)
monitors = ['loss_', 'y_hat', 'y']

rnn.run(X, Y, learn_alg, optimizer,
        monitors=monitors,
        update_interval=1,
        l2_reg=0.0001)

signals = [rnn.mons['loss_']]
fig = plot_filtered_signals(signals, y_lim=[-0.1, 1.5], plot_loss_benchmarks=True, filter_size=100)

i_hats = []
i_label = []
last_n = 1000
for i in range(last_n):
    
    i_hats.append(np.argmax(rnn.mons['y_hat'][-last_n+i]))
    if np.amax(rnn.mons['y'][-last_n+i])==1:
        i_label.append(np.argmax(rnn.mons['y'][-last_n+i]))
    else:
        i_label.append(-1)

acc = []
for i in range(last_n):
    if i_label!=-1:
        acc.append(i_hats[i]==i_label[i])
        
print(sum(acc)/len(acc))
        
        
        
        
        




