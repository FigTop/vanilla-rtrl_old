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

np.random.seed()

task = Coin_Task(2, 4, one_hot=True, deterministic=False)
data = task.gen_data(15000, 1000)
#task = Copy_Task(10, 3)

n_in     = task.n_in
n_hidden = 32
n_out    = task.n_out

W_in  = np.random.normal(0, np.sqrt(1/(n_in)), (n_hidden, n_in))
W_rec = np.random.normal(0, np.sqrt(1/(n_hidden)), (n_hidden, n_hidden))
W_rec = 0.54*np.eye(n_hidden)
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
SG_optimizer = SGD(lr=0.01, clipnorm=0.5)
learn_alg = DNI(rnn, SG_optimizer, monitors=['sg_loss', 'sg'], lambda_mix=0, l2_reg=0.001)
#learn_alg = KF_RTRL(rnn)
#comp_alg = BPTT(rnn, 1, 6, monitors=['credit_assignment'], use_historical_W=False)
monitors = ['loss_', 'alignment', 'y_hat', 'a', 'y', 'W_rec']

rnn.run(data,
        learn_alg=learn_alg,
        optimizer=optimizer,
        monitors=monitors,
        update_interval=1,
        l2_reg=0.001,
        check_accuracy=True),
        #comparison_alg=comp_alg)

W = np.nan_to_num(np.array(rnn.mons['W_rec']))
W_radii = get_spectral_radii(W)

task.plot_filtered_signals([W_radii], plot_loss_benchmarks=False)

if True:
    fig = plt.figure()
    signals = [rnn.mons['loss_'], learn_alg.mons['sg_loss']]
    task.plot_filtered_signals(signals, y_lim=[0, 1.5])
    plt.legend(['Loss', 'SG Loss'])
if False:
    fig = plt.figure()
    signals = [[a[i] for a in rnn.mons['alignment']] for i in range(3)]
    task.plot_filtered_signals(signals, y_lim=[-1.2, 1.2], plot_loss_benchmarks=False)
    plt.legend(['W_rec', 'W_in', 'b_rec'])
    plt.ylabel('Normalized Dot Product')



































