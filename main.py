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
from dynamics import *
import multiprocessing as mp
from functools import partial

if os.environ['HOME'] == '/home/oem214':
    n_seeds = 30
    try:
        i_job = int(os.environ['SLURM_ARRAY_TASK_ID']) - 1
    except KeyError:
        i_job = 0
    difficulties = [8, 12, 16, 20, 32, 64, 128, 256]
    T_horizons = list(range(10))
    #difficulties = [8, 16, 32, 64, 128]
    macro_configs = config_generator(algorithm=[None])
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

    #np.random.seed(1)

task = Flip_Flop_Task(3, 0.05)
data = task.gen_data(200000, 6000)

n_in = task.n_in
n_hidden = 128
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

optimizer = Stochastic_Gradient_Descent(lr=0.0001)
learn_alg = Efficient_BPTT(rnn, 10)

comp_algs = []
monitors = ['rnn.loss_', 'learn_alg.rec_grads-norm']

dynamics_analysis = Vanilla_PCA(2000, 3)

sim = Simulation(rnn)
sim.run(data, learn_alg=learn_alg, optimizer=optimizer,
        comp_algs=comp_algs,
        monitors=monitors,
        verbose=True,
        check_accuracy=False,
        check_loss=True,
        dynamics_analysis=dynamics_analysis)

test_sim = Simulation(rnn)
test_sim.run(data,
             mode='test',
             monitors=['rnn.loss_', 'rnn.y_hat', 'rnn.a'],
             verbose=False)

plt.figure()
plt.plot(test_sim.mons['rnn.y_hat'][:, 0])
plt.plot(data['test']['Y'][:, 0])

PC_results = np.array(sim.dynamics_analysis.results['analysis'])
Q = PC_results.reshape(PC_results.shape[0], -1)
M = Q.dot(Q.T)
plt.figure()
plt.imshow(M)
plt.figure()
plt.plot(np.diag(M[1:,:-1]))

#for i in range(0, 200, 20):
#    
#    test_sim = Simulation(dynamics_analysis.results['checkpoint_rnn'][i])
#    test_sim.run(data,
#                 mode='test',
#                 monitors=['rnn.a'],
#                 verbose=False)
#    ssa = State_Space_Analysis(test_sim.mons['rnn.a'], n_PCs=3)
#    ssa.plot_in_state_space(test_sim.mons['rnn.a'], '.', alpha=0.01)
#    
#filt = uniform_filter1d(sim.mons['rnn.loss_'], 1000)
#plt.figure()
#plt.plot(filt)
#for i in range(10):
#    plt.axvline(x=i*200000/10, color='k', linestyle='--')
    
#with open('notebooks/good_ones/first_try', 'wb') as f:
#    pickle.dump(sim, f)

if os.environ['HOME'] == '/Users/omarschall' and False:

    ssa = State_Space_Analysis(test_sim.mons['rnn.a'], n_PCs=3)
    ssa.plot_in_state_space(test_sim.mons['rnn.a'], '.', alpha=0.01)
    #ssa.fig.axes[0].set_xlim([-0.6, 0.6])
    #ssa.fig.axes[0].set_ylim([-0.6, 0.6])
    #ssa.fig.axes[0].set_zlim([-0.8, 0.8])
#    for i in range(8):
#        col = 'C{}'.format(i+1)
#        ssa.plot_in_state_space(A[i][:-1,:], color=col)
#        ssa.plot_in_state_space(A[i][-1,:].reshape((1,-1)), 'x', color=col)

if os.environ['HOME'] == '/home/oem214':

#    result = {'sim': sim, 'i_seed': i_seed, 'task': task,
#              'config': params, 'i_config': i_config, 'i_job': i_job,
#              'processed_data': processed_data}
    save_dir = os.environ['SAVEPATH']
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_path = os.path.join(save_dir, 'result_'+str(i_job))

    with open(save_path, 'wb') as f:
        pickle.dump(result, f)




























