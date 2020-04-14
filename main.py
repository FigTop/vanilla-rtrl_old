#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 16:30:58 2018

@author: omarschall
"""

import numpy as np
from network import *
from simulation import *
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
from state_space import *
from dynamics import *
import multiprocessing as mp
from functools import partial
from sklearn.cluster import DBSCAN

if os.environ['HOME'] == '/home/oem214':
    n_seeds = 1
    try:
        i_job = int(os.environ['SLURM_ARRAY_TASK_ID']) - 1
    except KeyError:
        i_job = 0
    macro_configs = config_generator(i_start=list(range(10000, 30000, 100)))
    micro_configs = tuple(product(macro_configs, list(range(n_seeds))))

    params, i_seed = micro_configs[i_job]
    i_config = i_job//n_seeds
    np.random.seed(i_job)

    save_dir = os.environ['SAVEPATH']
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
        
if os.environ['HOME'] == '/Users/omarschall':
    params = {'i_start': 13100}
    i_job = 0
    save_dir = '/Users/omarschall/vanilla-rtrl/library'

    #np.random.seed(1)

np.random.seed(0)
task = Flip_Flop_Task(3, 0.05, tau_task=1)
N_train = 100000
N_test = 10000
data = task.gen_data(N_train, N_test)
with open('notebooks/good_ones/bptt_fave', 'rb') as f:
    sim = pickle.load(f)
result = {}
for i_checkpoint in range(params['i_start'], params['i_start'] + 100, 10):
    analyze_checkpoint(sim.checkpoints[i_checkpoint], data, verbose=False,
                        sigma_pert=0.5, N=600, parallelize=False,
                        N_iters=8000, same_LR_criterion=7000)
    
    result['checkpoint_{}'.format(i_checkpoint)] = deepcopy(sim.checkpoints[i_checkpoint])
    
# np.random.seed(i_job)

# n_in = task.n_in
# n_hidden = 64
# n_out = task.n_out

# W_in  = np.random.normal(0, np.sqrt(1/(n_in)), (n_hidden, n_in))
# W_rec = np.linalg.qr(np.random.normal(0, 1, (n_hidden, n_hidden)))[0]
# W_out = np.random.normal(0, np.sqrt(1/(n_hidden)), (n_out, n_hidden))
# W_FB = np.random.normal(0, np.sqrt(1/n_out), (n_out, n_hidden))

# b_rec = np.zeros(n_hidden)
# b_out = np.zeros(n_out)

# alpha = 1

# rnn = RNN(W_in, W_rec, W_out, b_rec, b_out,
#           activation=tanh,
#           alpha=alpha,
#           output=identity,
#           loss=mean_squared_error)

# optimizer = SGD_Momentum(lr=0.0005, mu=0.6, clip_norm=0.3)
# if params['algorithm'] == 'E-BPTT':
#     learn_alg = Efficient_BPTT(rnn, 10, L2_reg=0.0001)
# elif params['algorithm'] == 'RFLO':
#     learn_alg = RFLO(rnn, alpha=alpha, L2_reg=0.0001)

# comp_algs = []
# monitors = ['rnn.a']

# sim = Simulation(rnn)
# sim.run(data, learn_alg=learn_alg, optimizer=optimizer,
#         comp_algs=comp_algs,
#         monitors=monitors,
#         verbose=True,
#         report_accuracy=False,
#         report_loss=True,
#         checkpoint_interval=list(range(10000, 30000, 10)))

# with open('notebooks/good_ones/bptt_fave', 'wb') as f:
#     pickle.dump(sim, f)

if os.environ['HOME'] == '/Users/omarschall':

    test_sim = Simulation(rnn)
    test_sim.run(data,
                  mode='test',
                  monitors=['rnn.loss_', 'rnn.y_hat', 'rnn.a'],
                  verbose=False)
    
    plt.figure()
    plt.plot(test_sim.mons['rnn.y_hat'][:, 0])
    plt.plot(data['test']['Y'][:, 0])
    plt.xlim([0, 1000])

if os.environ['HOME'] == '/home/oem214':

    # result = {'sim': sim, 'i_seed': i_seed, 'task': task,
    #           'config': params, 'i_config': i_config, 'i_job': i_job,
    #           'processed_data': processed_data}
    result['i_job'] = i_job
    result['config'] = params
    save_dir = os.environ['SAVEPATH']
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_path = os.path.join(save_dir, 'result_'+str(i_job))

    with open(save_path, 'wb') as f:
        pickle.dump(result, f)




























