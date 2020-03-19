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
from sklearn import linear_model
from state_space import State_Space_Analysis

if os.environ['HOME'] == '/home/oem214':
    n_seeds = 12
    try:
        i_job = int(os.environ['SLURM_ARRAY_TASK_ID']) - 1
    except KeyError:
        i_job = 0
    difficulties = [8, 12, 16, 20, 32, 64, 128, 256]
    T_horizons = list(range(10))
    #difficulties = [8, 16, 32, 64, 128]
    macro_configs = config_generator(algorithm=['BPTT'],
                                     difficulty=difficulties,
                                     T_horizon=T_horizons)
#    macro_configs = config_generator(alpha=[0.5, 1],
#                                     difficulty=difficulties)
#algorithm=['Only_Output_Weights', 'RTRL',
#                                               'UORO', 'KF-RTRL', 'R-KF-RTRL',
#                                               'BPTT', 'DNI', 'DNIb',
#                                               'RFLO', 'KeRNL'],
    micro_configs = tuple(product(macro_configs, list(range(n_seeds))))

    params, i_seed = micro_configs[i_job]
    i_config = i_job//n_seeds
    np.random.seed(i_job)

    save_dir = os.environ['SAVEPATH']
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

if os.environ['HOME'] == '/Users/omarschall':
    params = {'algorithm': 'RTRL'}
    i_job = 0
    save_dir = '/Users/omarschall/vanilla-rtrl/library'

    np.random.seed(1)

task = Flip_Flop_Task(3, 0.5)
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
          output=identity,
          loss=mean_squared_error)

optimizer = Stochastic_Gradient_Descent(lr=0.001)
SG_optimizer = Stochastic_Gradient_Descent(lr=0.001)

if params['algorithm'] == 'Only_Output_Weights':
    learn_alg = Only_Output_Weights(rnn)
if params['algorithm'] == 'RTRL':
    learn_alg = RTRL(rnn, L2_reg=0.01)
if params['algorithm'] == 'UORO':
    learn_alg = UORO(rnn)
if params['algorithm'] == 'KF-RTRL':
    learn_alg = KF_RTRL(rnn)
if params['algorithm'] == 'R-KF-RTRL':
    learn_alg = Reverse_KF_RTRL(rnn)
if params['algorithm'] == 'BPTT':
    learn_alg = Future_BPTT(rnn, params['T_horizon'])
if params['algorithm'] == 'DNI':
    learn_alg = DNI(rnn, SG_optimizer)
if params['algorithm'] == 'DNIb':
    J_lr = 0.001
    learn_alg = DNI(rnn, SG_optimizer, use_approx_J=True, J_lr=J_lr,
                    SG_label_activation=tanh, W_FB=W_FB)
    learn_alg.name = 'DNIb'
if params['algorithm'] == 'RFLO':
    learn_alg = RFLO(rnn, alpha=alpha)
if params['algorithm'] == 'KeRNL':
    sigma_noise = 0.0000001
    base_learning_rate = 0.01
    kernl_lr = base_learning_rate/sigma_noise
    KeRNL_optimizer = Stochastic_Gradient_Descent(kernl_lr)
    learn_alg = KeRNL(rnn, KeRNL_optimizer, sigma_noise=sigma_noise)

comp_algs = []
monitors = []

sim = Simulation(rnn)
sim.run(data, learn_alg=learn_alg, optimizer=optimizer,
        comp_algs=comp_algs,
        monitors=monitors,
        verbose=True,
        check_accuracy=False,
        check_loss=True)

if os.environ['HOME'] == '/Users/omarschall':

    #Test run
    np.random.seed(2)
    n_test = 10000
    data = task.gen_data(0, n_test)
    test_sim = deepcopy(sim)
    test_sim.run(data,
                 mode='test',
                 monitors=['rnn.loss_', 'rnn.y_hat', 'rnn.a'],
                 verbose=False)
    fig = plt.figure()
    
    plt.plot(test_sim.mons['rnn.y_hat'][:,0])
    plt.plot(data['test']['Y'][:,0])
    plt.xlim([9800, 10000])

    plt.figure()
    x = test_sim.mons['rnn.y_hat'].flatten()
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




























