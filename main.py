#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 16:30:58 2018

@author: omarschall
"""

import numpy as np
from network import RNN
from network import Noisy_RNN
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
from dynamics import *
import multiprocessing as mp
from functools import partial

if os.environ['HOME'] == '\\Users\\colin':
    params = {}
    i_job = 0
    save_dir = '\\Users\\colin\\Documents\\Python\\vanilla-rtrl'
elif os.environ['HOME'] == '/home/oem214':
    n_seeds = 100
    try:
        i_job = int(os.environ['SLURM_ARRAY_TASK_ID']) - 1
    except KeyError:
        i_job = 0
    macro_configs = config_generator(mu=[0, 0.3, 0.6],
                                     clip_norm=[0.1, 0.3],
                                     L2_reg=[0.00001])
    micro_configs = tuple(product(macro_configs, list(range(n_seeds))))

    params, i_seed = micro_configs[i_job]
    i_config = i_job//n_seeds
    np.random.seed(i_job + 20)

    save_dir = os.environ['SAVEPATH']
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

elif os.environ['HOME'] == '/Users/omarschall':
    params = {}
    i_job = 0
    save_dir = '/Users/omarschall/vanilla-rtrl/library'

    #np.random.seed(1)

#task = Flip_Flop_Task(2, 0.1, tau_task=1)
task = Add_Task(4,6, deterministic = True)
data = task.gen_data(200000, 1000)

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

rnn = Noisy_RNN(W_in, W_rec, W_out, b_rec, b_out,
          activation=tanh,
          alpha=alpha,
          output=softmax,
          loss=softmax_cross_entropy)

sigma = 0.5/np.sqrt(2)
optimizer = SGD_Momentum(lr=0.001, mu = 0.6)
#learn_alg = REINFORCE(rnn, sigma = sigma, decay = 0.15, loss_decay = 0.01)
learn_alg = REINFORCE_RFLO(rnn, sigma = sigma, decay = 0.15, loss_decay = 0.01)

comp_algs = []
#monitors = ['learn_alg.rec_grads-norm', 'rnn.loss_']
#monitors = ['learn_alg.e_trace-norm']
monitors = []

sim = Simulation(rnn)
sim.run(data, learn_alg=learn_alg, optimizer=optimizer,
        comp_algs=comp_algs,
        monitors=monitors,
        verbose=True,
        report_accuracy=False,
        report_loss=True,
        sigma = sigma,
        checkpoint_interval=None)


test_sim = Simulation(rnn)
test_sim.run(data,
             sigma = sigma,
              mode='test',
              monitors=['rnn.loss_', 'rnn.y_hat', 'rnn.a'],
              verbose=False)

plt.figure()
plt.plot(test_sim.mons['rnn.y_hat'][:, 0])
plt.plot(data['test']['X'][:, 0])
plt.plot(data['test']['Y'][:, 0])
plt.xlim([0, 100])

plt.figure()
filtered_loss = uniform_filter1d(test_sim.mons['rnn.loss_'], 100)
plt.plot(filtered_loss)

if False and os.environ['HOME'] == '/Users/omarschall':

    ssa = State_Space_Analysis(test_sim.mons['rnn.a'], n_PCs=3)
    ssa.clear_plot()
    ssa.plot_in_state_space(test_sim.mons['rnn.a'], '.', alpha=0.01)
    #ssa.plot_in_state_space(a_values, 'x', alpha=0.5)
    ssa.plot_in_state_space(result['a_trajectory'], color='C6')
    ssa.plot_in_state_space(result['a_trajectory'][-2:], 'o', color='C6')
    x_sizes = 1/np.sqrt(KEs)
    for i, x_size in enumerate(sizes):
        ssa.plot_in_state_space(a_values[i].reshape(1, -1), 'x', color='C1', alpha=0.2, markersize=x_size)
    #ssa.fig.axes[0].set_xlim([-0.6, 0.6])
    #ssa.fig.axes[0].set_ylim([-0.6, 0.6])
    #ssa.fig.axes[0].set_zlim([-0.8, 0.8])
#    for i in range(8):
#        col = 'C{}'.format(i+1)
#        ssa.plot_in_state_space(A[i][:-1,:], color=col)
#        ssa.plot_in_state_space(A[i][-1,:].reshape((1,-1)), 'x', color=col)

if False and os.environ['HOME'] == '/home/oem214':

    result = {'sim': sim, 'i_seed': i_seed, 'task': task,
              'config': params, 'i_config': i_config, 'i_job': i_job,
              'processed_data': processed_data}
    save_dir = os.environ['SAVEPATH']
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_path = os.path.join(save_dir, 'result_'+str(i_job))

    with open(save_path, 'wb') as f:
        pickle.dump(result, f)




























