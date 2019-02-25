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
from itertools import product
import os
import pickle
from copy import copy
from state_space import State_Space_Analysis
from pdb import set_trace
from scipy.stats import linregress

if os.environ['HOME']=='/home/oem214':
    n_seeds = 10
    try:
        i_job = int(os.environ['SLURM_ARRAY_TASK_ID']) - 1
    except KeyError:
        i_job = 0
    macro_configs = config_generator(alpha_RFLO=[0.2, 0.3, 0.4],
                                     lr=[0.0005, 0.001])
    micro_configs = tuple(product(macro_configs, list(range(n_seeds))))
    
    params, i_seed = micro_configs[i_job]
    i_config = i_job//n_seeds
    np.random.seed(i_seed)

if os.environ['HOME']=='/Users/omarschall':
    params = {'lr': 0.001,
              'W_a_lr': 0.01,
              'alpha': 0.3}

task = Coin_Task(4, 6, one_hot=True, deterministic=True, tau_task=4)
data = task.gen_data(1000000, 8000)

n_in     = task.n_in
n_hidden = 32
n_out    = task.n_out

W_in  = np.random.normal(0, np.sqrt(1/(n_in)), (n_hidden, n_in))
W_rec = np.linalg.qr(np.random.normal(0, 1, (n_hidden, n_hidden)))[0]
W_out = np.random.normal(0, np.sqrt(1/(n_hidden)), (n_out, n_hidden))
W_FB = np.random.normal(0, np.sqrt(1/n_out), (n_out, n_hidden))

b_rec = np.zeros(n_hidden)
b_out = np.zeros(n_out)

alpha = 0.3

rnn = RNN(W_in, W_rec, W_out, b_rec, b_out,
          activation=tanh,
          alpha=alpha,
          output=softmax,
          loss=softmax_cross_entropy)

optimizer = SGD(lr=params['lr'])#, clipnorm=5)
#SG_optimizer = SGD(lr=0.01)
#learn_alg = DNI(rnn, SG_optimizer, W_a_lr=params['W_a_lr'], backprop_weights='exact',
#                SG_label_activation=identity)#, W_FB=W_FB)
                #train_SG_with_exact_CA=False)
#learn_alg.SG_init(8)
#learn_alg = RTRL(rnn)
#T = 20
#comp_alg = Forward_BPTT(rnn, T)
learn_alg = RFLO(rnn, alpha=params['alpha_RFLO'], W_FB=W_FB)
#cCclearn_alg = BPTT(rnn, 1, 10)
#monitors = ['loss_', 'y_hat', 'sg_loss', 'loss_a', 'sg_target-norm', 'global_grad-norm', 'A-norm', 'a-norm']
#monitors += ['CA_forward_est', 'CA_SG_est']
monitors = ['loss_', 'y_hat']#, 'sg_loss', 'loss_a', 'sg', 'CA', 'W_rec_alignment']

sim = Simulation(rnn, learn_alg, optimizer, L2_reg=0.00001)#, comparison_alg=comp_alg)
sim.run(data,
        monitors=monitors,
        verbose=True)
        #est_CA_interval=5000,
        #save_model_interval=5000)

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
    fig2 = plot_filtered_signals(signals, filter_size=100, y_lim=[0, 20])
    plt.legend(legend)
    
    try:
        plt.figure()
        dots = (sim.mons['sg'][:-(T-1)]*sim.mons['CA']).sum(1)
        norms = np.sqrt(np.square(sim.mons['sg'][:-(T-1)]).sum(1)*np.square(sim.mons['CA']).sum(1))
        plt.plot(dots/norms, '.', alpha=0.3)
    except ValueError:
        pass
    
    #Test run
    n_test = 10000
    data = task.gen_data(10, n_test)
    test_sim = Simulation(sim.net, learn_alg=None, optimizer=None)
    test_sim.run(data, mode='test', monitors=['loss_', 'y_hat', 'a'])
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

    result = {'sim': sim, 'i_seed': i_seed, 'task': task,
              'config': params, 'i_config': i_config, 'i_job': i_job}
    save_dir = os.environ['SAVEPATH']
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_path = os.path.join(save_dir, 'rnn_'+str(i_job))
    
    with open(save_path, 'wb') as f:
        pickle.dump(result, f)




























