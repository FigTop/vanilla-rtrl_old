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
try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    pass
import time
from optimizers import *
from analysis_funcs import *
from learning_algorithms import *
import os
import pickle

try:
    i_job = int(os.environ['SLURM_ARRAY_TASK_ID']) - 1
except KeyError:
    i_job = np.random.randint(1000)

i_seed = i_job
#i_seed = 471
np.random.seed(i_seed)

task = Coin_Task(3, 5, one_hot=True, deterministic=False)
data = task.gen_data(20000, 10000)
#task = Copy_Task(10, 3)

n_in     = task.n_in
n_hidden = 32
n_out    = task.n_out

W_in  = np.random.normal(0, np.sqrt(1/(n_in)), (n_hidden, n_in))
#W_rec = np.random.normal(0, np.sqrt(1/(2*n_hidden)), (n_hidden, n_hidden))
W_rec = np.linalg.qr(np.random.normal(0, 1, (n_hidden, n_hidden)))[0]
W_out = np.random.normal(0, np.sqrt(1/(n_hidden)), (n_out, n_hidden))

b_rec = np.zeros(n_hidden)
b_out = np.zeros(n_out)

alpha = 1

rnn = RNN(W_in, W_rec, W_out, b_rec, b_out,
          activation=tanh,
          alpha=alpha,
          output=softmax,
          loss=softmax_cross_entropy)

optimizer = SGD(lr=0.001, clipnorm=1.0)
SG_optimizer = SGD(lr=0.01)
learn_alg = DNI(rnn, SG_optimizer, activation=tanh,
                monitors=['sg_loss', 'sg_norm', 'q'],
                lambda_mix=0, l2_reg=0, fix_SG_interval=5)
#learn_alg = BPTT(rnn, 1, 10)
monitors = ['loss_', 'a', 'W_rec']
#            'W_radius',
#            'A_radius']

rnn.run(data,
        learn_alg=learn_alg,
        optimizer=optimizer,
        monitors=monitors,
        update_interval=1,
        l2_reg=0.01,
        check_accuracy=True,
        verbose=True)


if os.environ['HOME']=='/Users/omarschall':

    
    signals = [rnn.mons['loss_'], rnn.learn_alg.mons['sg_loss']]#, rnn.mons['W_radius'], rnn.mons['A_radius']]
    signals2 = [(learn_alg.mons['q']**2).mean(1)]
#                (rnn.mons['W_rec']**2).mean(1).mean(1),
#                (learn_alg.mons['A']**2).mean(1).mean(1),
#                (learn_alg.mons['B']**2).mean(1).mean(1),
#                (learn_alg.mons['C']**2).mean(1)]
    fig1 = plot_filtered_signals(signals, filter_size=1000, y_lim=[0, 1.5])
#    plt.legend(['Loss', 'SG Loss'])#, 'W_rec alignment'])
    fig2 = plot_filtered_signals(signals2, filter_size=1000, plot_loss_benchmarks=False)
    #plt.legend(['||a||', '||W_rec||', '||A||', '||B||', '||C||'])

if os.environ['HOME']=='/home/oem214':

    result = {'rnn': rnn, 'i_seed': i_seed}
    save_dir = os.environ['SAVEPATH']
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_path = os.path.join(save_dir, 'rnn_'+str(i_job))
    
    with open(save_path, 'wb') as f:
        pickle.dump(result, f)




























