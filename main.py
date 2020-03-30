#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 16:30:58 2018

@author: omarschall
"""

import numpy as np
from network import RNN
from simulation import Simulation
from gen_data import *
try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    pass
from optimizers import *
from analysis_funcs import *
from learning_algorithms import *
from functions import *
from itertools import product
import os
import pickle
from copy import deepcopy
from scipy.ndimage.filters import uniform_filter1d

if os.environ['HOME'] == '/home/oem214':
    n_seeds = 15
    try:
        i_job = int(os.environ['SLURM_ARRAY_TASK_ID']) - 1
    except KeyError:
        i_job = 0
    macro_configs = config_generator(algorithm=['Only_Output_Weights', 'RTRL',
                                               'UORO', 'KF-RTRL', 'R-KF-RTRL',
                                               'BPTT', 'DNI', 'DNIb',
                                               'RFLO', 'KeRNL'],
                                     difficulty=[28, 49, 56, 98, 112, 196])
    micro_configs = tuple(product(macro_configs, list(range(n_seeds))))

    params, i_seed = micro_configs[i_job]
    i_config = i_job//n_seeds
    np.random.seed(i_job)

    save_dir = os.environ['SAVEPATH']
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

if os.environ['HOME'] == '/Users/omarschall':
    params = {}
    i_job = 0
    save_dir = '/Users/omarschall/vanilla-rtrl/library'

    #np.random.seed()

task = Add_Task(n_1, n_2, deterministic=True, tau_task=1)
data = task.gen_data(100000, 5000)

n_in = task.n_in
n_hidden = 32
n_out = task.n_out

W_in  = np.random.normal(0, np.sqrt(1/(n_in)), (n_hidden, n_in))
W_rec = np.linalg.qr(np.random.normal(0, 1, (n_hidden, n_hidden)))[0]
W_out = np.random.normal(0, np.sqrt(1/(n_hidden)), (n_out, n_hidden))
W_FB = np.random.normal(0, np.sqrt(1/n_out), (n_out, n_hidden))
b_rec = np.zeros(n_hidden)
b_out = np.zeros(n_out)

alpha = 1
rnn = RNN(W_in, W_rec, W_out, b_rec, b_out,
          activation=tanh,
          alpha=alpha,
          output=softmax,
          loss=softmax_cross_entropy)

optimizer = Stochastic_Gradient_Descent(lr=0.001)
learn_alg = Only_Output_Weights(rnn)

monitors = ['net.loss_', 'net.y_hat', 'net.sigma']

sim = Simulation(rnn)
sim.run(data, learn_alg=learn_alg, optimizer=optimizer,
        comp_algs=comp_algs,
        monitors=monitors,
        verbose=True,
        check_accuracy=False,
        check_loss=True,
        sigma=0.07)

if os.environ['HOME'] == '/Users/omarschall':

    #plot_filtered_signals([sim.mons['net.loss_']])

    #Test run
    np.random.seed(2)
    n_test = 10000
    data = task.gen_data(100, n_test)
    test_sim = deepcopy(sim)
    test_sim.run(data,
                 mode='test',
                 monitors=['net.loss_', 'net.y_hat', 'net.a'],
                 verbose=False)
    fig = plt.figure()
    for i in range(2):
        plt.plot(test_sim.mons['net.y_hat'][:,i], color='C{}'.format(i))
        plt.plot(data['test']['Y'][:,i], color='C{}'.format(i), linestyle='--')
        plt.xlim([9800, 10000])

    plt.figure()
    x = test_sim.mons['net.y_hat'].flatten()
    y = data['test']['Y'].flatten()
    plt.plot(x, y, '.', alpha=0.05)
    plt.plot([np.amin(x), np.amax(x)],
              [np.amin(y), np.amax(y)], 'k', linestyle='--')
    plt.axis('equal')

if os.environ['HOME'] == '/home/oem214':

    result = {'sim': sim, 'i_seed': i_seed, 'task': task,
              'config': params, 'i_config': i_config, 'i_job': i_job,
              'processed_data': processed_data}
    save_dir = os.environ['SAVEPATH']
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_path = os.path.join(save_dir, 'rnn_'+str(i_job))

    with open(save_path, 'wb') as f:
        pickle.dump(result, f)




























