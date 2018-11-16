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
from copy import copy

try:
    i_job = int(os.environ['SLURM_ARRAY_TASK_ID']) - 1
except KeyError:
    i_job = np.random.randint(1000)

i_seed = i_job
#i_seed = 339
np.random.seed(i_seed)

configs = [[3, 4, 5], ['symmetric', 'random'], ['exact', 'approximate']]

confs = [np.random.choice(conf) for conf in configs]
confs = [3, 'random', 'approximate']

n_back = confs[0]
fb = confs[1]
bp_w = confs[2]

task = Coin_Task(n_back, n_back+2, one_hot=True, deterministic=False)
#task = Copy_Task(5, 2)
data = task.gen_data(40000, 1000)

n_in     = task.n_in
n_hidden = 32
n_out    = task.n_out

W_in  = np.random.normal(0, np.sqrt(1/(n_in)), (n_hidden, n_in))
#W_rec = np.random.normal(0, np.sqrt(1/(n_hidden)), (n_hidden, n_hidden))
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

optimizer = SGD(lr=0.001)
SG_optimizer = SGD(lr=0.01)
learn_alg = DNI(rnn, SG_optimizer, activation=identity,
                monitors=['sg_loss', 'loss_a'],
                l2_reg=0, fix_SG_interval=5,
                W_a_lr=0.01, SG_label_activation=tanh,
                feedback=fb, backprop_weights=bp_w)
monitors = ['loss_', 'acc', 'y_hat']

rnn.run(data,
        learn_alg=learn_alg,
        optimizer=optimizer,
        monitors=monitors,
        update_interval=1,
        l2_reg=0.0001,
        check_accuracy=True,
        verbose=True)

if os.environ['HOME']=='/Users/omarschall':

    
    signals = [rnn.mons['loss_'], rnn.learn_alg.mons['sg_loss'], rnn.learn_alg.mons['loss_a']]
    fig1 = plot_filtered_signals(signals, filter_size=1000, y_lim=[0, 1.5])

if os.environ['HOME']=='/home/oem214':

    result = {'rnn': rnn, 'i_seed': i_seed, 'config': confs}
    save_dir = os.environ['SAVEPATH']
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_path = os.path.join(save_dir, 'rnn_'+str(i_job))
    
    with open(save_path, 'wb') as f:
        pickle.dump(result, f)




























