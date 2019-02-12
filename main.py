#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 16:30:58 2018

@author: omarschall
"""

import numpy as np
from network import RNN
from fast_weights_network import Fast_Weights_RNN
from simulation import Simulation
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
from functions import *
import os
import pickle
from copy import copy

try:
    i_job = int(os.environ['SLURM_ARRAY_TASK_ID']) - 1
except KeyError:
    i_job = np.random.randint(1000)

#taus = [2, 3, 4]
#alphas  = [1, 0.7, 0.5, 0.3]
seeds = list(range(20))
#HPs = sum(sum([[[[a, t, s] for t in taus] for a in alphas] for s in seeds],[]), [])
#alpha, tau, i_seed = HPs[0]

i_seed = i_job
np.random.seed(i_seed)

task = Coin_Task(4, 6, one_hot=True, deterministic=True, tau_task=4)
#task = Sine_Wave(0.001, [0.01, 0.007, 0.003, 0.001], amplitude=0.1, method='regular')
data = task.gen_data(200000, 5000)

n_in     = task.n_in
n_hidden = 32
n_out    = task.n_out

W_in  = np.random.normal(0, np.sqrt(1/(n_in)), (n_hidden, n_in))
#W_rec = np.random.normal(0, np.sqrt(1/(n_hidden)), (n_hidden, n_hidden))
W_rec = np.linalg.qr(np.random.normal(0, 1, (n_hidden, n_hidden)))[0]
W_out = np.random.normal(0, np.sqrt(1/(n_hidden)), (n_out, n_hidden))
W_FB = np.random.normal(0, np.sqrt(1/n_out), (n_out, n_hidden))

b_rec = np.zeros(n_hidden)
b_out = np.zeros(n_out)

A = np.zeros_like(W_rec)
n_S = 10

alpha = 0.3

rnn = RNN(W_in, W_rec, W_out, b_rec, b_out,
          activation=tanh,
          alpha=alpha,
          output=softmax,
          loss=softmax_cross_entropy)

#rnn = Fast_Weights_RNN(W_in, W_rec, W_out, b_rec, b_out,
#                       activation=tanh,
#                       alpha=alpha,
#                       output=softmax,
#                       loss=softmax_cross_entropy,
#                       A=A, lmbda=0.95, eta=0.5, n_S=10)

optimizer = SGD(lr=0.0005)#, lr_decay_rate=0.9999, min_lr=0.00005)#, clipnorm=1.0)
SG_optimizer = SGD(lr=0.01)
learn_alg = DNI(rnn, SG_optimizer, W_a_lr=0.01, backprop_weights='approximate',
                SG_label_activation=tanh, W_FB=W_FB)#, SG_target_clipnorm=1)#, W_FB=W_FB)
#learn_alg = KF_RTRL(rnn, P0=0.8, P1=1.3)
#learn_alg = UORO(rnn, epsilon=1e-10)
comp_alg = RTRL(rnn)
#learn_alg = RFLO(rnn, alpha=alpha, W_FB=W_FB)
#learn_alg = BPTT(rnn, 1, 20)
#monitors = ['loss_', 'a', 'y_hat', 'sg_loss', 'loss_a']
monitors = ['loss_', 'y_hat', 'sg_loss', 'loss_a', 'W_rec_alignment']

sim = Simulation(rnn, learn_alg, optimizer, L2_reg=0.0001, comparison_alg=comp_alg)
sim.run(data,
        monitors=monitors,
        verbose=True,
        check_loss=True,
        test_loss_thr=0.52,
        report_interval=2000)

if os.environ['HOME']=='/Users/omarschall':

    signals = []
    legend = []
    for key in sim.mons.keys():
        s = sim.mons[key].shape
        if len(s)==1 and s[0]>0:
            signals.append(sim.mons[key])
            legend.append(key)
    fig1 = plot_filtered_signals(signals, filter_size=100, y_lim=[0, 1])
    plt.legend(legend)
    
    #Test run
    test_sim = Simulation(rnn, learn_alg=None, optimizer=None)
    test_sim.run(data, mode='test', monitors=['loss_', 'y_hat', 'sg_loss', 'loss_a'])
    plt.figure()
    plt.plot(test_sim.mons['y_hat'][:,0])
    plt.plot(data['test']['Y'][:,0])
    plt.ylim([0, 1.2])
    plt.xlim([3000, 3100])
    
    
    plt.figure()
    plt.plot(test_sim.mons['y_hat'][:,0],data['test']['Y'][:,0], '.', alpha=0.05)
    plt.plot([0, 1], [0, 1], 'k', linestyle='--')
    #plt.axis('equal')
    plt.ylim([0, 1.1])
    plt.xlim([0, 1.1])

if os.environ['HOME']=='/home/oem214':

    result = {'sim': sim, 'i_seed': i_seed, 'task': task}
    save_dir = os.environ['SAVEPATH']
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_path = os.path.join(save_dir, 'rnn_'+str(i_job))
    
    with open(save_path, 'wb') as f:
        pickle.dump(result, f)




























