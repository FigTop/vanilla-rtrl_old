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

LRs = [0.005, 0.001, 0.0005, 0.00001]
Ps  = [0.01, 0.003, 0.001, 0.0003, 0.0001, 0.00003]
HPs = sum([[[p, l] for l in LRs] for p in Ps],[])
p, lr = HPs[i_job]

#i_seed = i_job
i_seed = 1
np.random.seed(i_seed)
#task = Coin_Task(4, 6, one_hot=True, deterministic=False)
task = Sine_Wave(p, [0.001, 0.003, 0.001, 0.0003], amplitude=0.1, method='regular')
data = task.gen_data(4000000, 5000)

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

alpha = 0.03

rnn = RNN(W_in, W_rec, W_out, b_rec, b_out,
          activation=tanh,
          alpha=alpha,
          output=identity,
          loss=mean_squared_error)

#rnn = Fast_Weights_RNN(W_in, W_rec, W_out, b_rec, b_out,
#                       activation=tanh,
#                       alpha=alpha,
#                       output=softmax,
#                       loss=softmax_cross_entropy,
#                       A=A, lmbda=0.95, eta=0.5, n_S=10)

optimizer = SGD(lr=lr)#, clipnorm=1.0)
#SG_optimizer = SGD(lr=0.001)
#learn_alg = DNI(rnn, SG_optimizer, W_a_lr=0.001, backprop_weights='approximate',
#                SG_label_activation=tanh, W_FB=W_FB)
#learn_alg = KF_RTRL(rnn, P0=0.8, P1=1.3)
#learn_alg = UORO(rnn, epsilon=1e-10)
learn_alg = RTRL(rnn)
#learn_alg = RFLO(rnn, alpha=alpha, W_FB=W_FB)
#learn_alg = BPTT(rnn, 1, 20)
#monitors = ['loss_', 'a', 'y_hat', 'sg_loss', 'loss_a']
monitors = ['loss_', 'y_hat']#, 'sg_loss', 'loss_a']

sim = Simulation(rnn, learn_alg, optimizer, l2_reg=0.00001, sigma=0.0001)#, comparison_alg=comp_alg)
sim.run(data,
        monitors=monitors,
        verbose=True,
        check_loss=True)

if os.environ['HOME']=='/Users/omarschall':

    signals1 = [sim.mons['loss_']]#, sim.mons['sg_loss'], sim.mons['loss_a']]
    fig1 = plot_filtered_signals(signals1, filter_size=100, y_lim=[0, 1])
    plt.legend(['Loss'])
    
    #Test run
    sim = Simulation(rnn, learn_alg=None, optimizer=None)
    sim.run(data, mode='test', monitors=['loss_', 'y_hat'])
    plt.figure()
    plt.plot(sim.mons['y_hat'][:,0])
    plt.plot(data['test']['Y'][:,0])
    plt.ylim([-0.15, 0.15])
    #plt.title('RFLO on (4,6)-back task')

if os.environ['HOME']=='/home/oem214':

    result = {'sim': sim, 'i_seed': i_seed, 'task': task}
    save_dir = os.environ['SAVEPATH']
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_path = os.path.join(save_dir, 'rnn_'+str(i_job))
    
    with open(save_path, 'wb') as f:
        pickle.dump(result, f)




























