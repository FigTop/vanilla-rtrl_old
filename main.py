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
from netcomp.distance import netsimile

if os.environ['HOME'] == '/home/oem214':
    n_seeds = 1
    try:
        i_job = int(os.environ['SLURM_ARRAY_TASK_ID']) - 1
    except KeyError:
        i_job = 0
    #macro_configs = config_generator(i_start=list(range(0, 200000, 1000)))
    macro_configs = config_generator(algorithm=[None])
    micro_configs = tuple(product(macro_configs, list(range(n_seeds))))

    params, i_seed = micro_configs[i_job]
    i_config = i_job//n_seeds
    np.random.seed(i_job)

    save_dir = os.environ['SAVEPATH']
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

if os.environ['HOME'] == '/Users/omarschall':
    params = {'algorithm': 'E-BPTT', 'i_start': 45000}
    i_job = 0
    save_dir = '/Users/omarschall/vanilla-rtrl/library'

    #np.random.seed(1)

# result = {}

np.random.seed(0)
task = Flip_Flop_Task(1, 0.05, tau_task=1)
#task = Sine_Wave(0.05, [7, 12], method='regular')
N_train = 1000
N_test = 10000
data = task.gen_data(N_train, N_test)
#big_data = task.gen_data(100, 100000)


n_in = task.n_in
n_hidden = 64
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

optimizer = SGD_Momentum(lr=0.005, mu=0.6, clip_norm=0.3)
if params['algorithm'] == 'E-BPTT':
    learn_alg = Efficient_BPTT(rnn, 10, L2_reg=0.0001)
elif params['algorithm'] == 'RFLO':
    learn_alg = RFLO(rnn, alpha=alpha, L2_reg=0.0001, L1_reg=0.0001)
#learn_alg = Only_Output_Weights(rnn)

comp_algs = []
monitors = []

sim = Simulation(rnn)
sim.run(data, learn_alg=learn_alg, optimizer=optimizer,
        comp_algs=comp_algs,
        monitors=monitors,
        verbose=True,
        report_accuracy=False,
        report_loss=True,
        checkpoint_interval=10)

for i_checkpoint in range(0, 300, 10):
    checkpoint = sim.checkpoints[i_checkpoint]
    analyze_checkpoint(checkpoint, data, verbose=False,
                       sigma_pert=0.5, N=200, parallelize=True, n_PCs=2,
                       N_iters=3000, same_LR_criterion=2000)
    get_graph_structure(checkpoint)
    #train_VAE(checkpoint, big_data, T=10, latent_dim=256, lr=0.001)

# with open('notebooks/good_ones/1d_friend_2', 'wb') as f:
#     pickle.dump(sim, f)
with open('notebooks/good_ones/1d_friend_2', 'rb') as f:
    sim = pickle.load(f)
    
# big_data = task.gen_data(100, 500000)
# result = {}

# for i_checkpoint in range(params['i_start'], params['i_start'] + 1000, 100):
#     train_VAE(sim.checkpoints[i_checkpoint], big_data, T=10,
#               latent_dim=256, lr=0.001)
    
#     analyze_checkpoint(sim.checkpoints[i_checkpoint], data, verbose=False,
#                     sigma_pert=0.5, N=600, parallelize=False,
#                     N_iters=8000, same_LR_criterion=7000)
    
#     get_graph_structure(sim.checkpoints[i_checkpoint], parallelize=False)
    
#     result['checkpoint_{}'.format(i_checkpoint)] = deepcopy(sim.checkpoints[i_checkpoint])

#get_graph_structure(sim.checkpoints[N_train - 1])



# train_VAE(sim.checkpoints[499999], big_data, T=10, latent_dim=128, lr=0.001)

# plot_checkpoint_results(sim.checkpoints[0], data, plot_test_points=True,
#                         plot_cluster_means=True)

# #     #result['checkpoint_{}'.format(i_checkpoint)] = deepcopy(sim.checkpoints[i_checkpoint])

for i in range(180, 190, 10):
    ssa = plot_checkpoint_results(sim.checkpoints[i], data,
                                  plot_cluster_means=True,
                                  plot_fixed_points=True,
                                  plot_test_points=True,
                                  plot_graph_structure=True,
                                  n_vae_samples=None,
                                  n_test_samples=None)


W_recs = [sim.checkpoints[i]['rnn'].W_rec for i in range(0, 240, 10)]
W_in_norms = [norm(sim.checkpoints[i]['rnn'].W_in) for i in range(0, 240, 10)]
plot_eigenvalues(sim.checkpoints[40]['rnn'].W_rec, sim.checkpoints[50]['rnn'].W_rec)
plt.legend(['', 'Chkpt 5', 'Chkpt 6'])

W_recs = [sim.checkpoints[i]['rnn'].W_rec.flatten() for i in range(0, 240, 10)]
W_recs = np.array(W_recs)
Delta_W = W_recs[1:] - W_recs[:-1]

# with open('notebooks/good_ones/{}_net_prezzy'.format(params['algorithm']), 'wb') as f:
#     pickle.dump(sim, f)

if os.environ['HOME'] == '/Users/omarschall':

    # plt.figure()
    # n_filter = 2000
    # filtered_loss = uniform_filter1d(sim.mons['rnn.loss_'], n_filter)
    # rec_grad_norms = uniform_filter1d(sim.mons['learn_alg.rec_grads-norm'], n_filter)
    # rec_grad_norms *= (np.amax(filtered_loss) / np.amax(rec_grad_norms))
    # plt.plot(filtered_loss)
    # plt.plot(rec_grad_norms)
    # plt.xticks(list(range(0, 100000, 10000)))

    #plt.figure()
    #plt.plot(sim.mons['rnn.loss_'], sim.mons['learn_alg.rec_grads-norm'], '.', alpha=0.08)

    rnn = sim.checkpoints[N_train - 1]['rnn']
    #rnn = sim.checkpoints[99999]['rnn']
    test_sim = Simulation(rnn)
    test_sim.run(data,
                  mode='test',
                  monitors=['rnn.loss_', 'rnn.y_hat', 'rnn.a'],
                  verbose=False)

    plt.figure()
    plt.plot(data['test']['X'][:, 0] + 2.5, (str(0.6)), linestyle='--')
    plt.plot(data['test']['Y'][:, 0] + 2.5, 'C0')
    plt.plot(test_sim.mons['rnn.y_hat'][:, 0] + 2.5, 'C3')
    plt.plot(data['test']['X'][:, 1], (str(0.6)), linestyle='--')
    plt.plot(data['test']['Y'][:, 1], 'C0')
    plt.plot(test_sim.mons['rnn.y_hat'][:, 1], 'C3')
    plt.plot(data['test']['X'][:, 2] - 2.5, (str(0.6)), linestyle='--')
    plt.plot(data['test']['Y'][:, 2] - 2.5, 'C0')
    plt.plot(test_sim.mons['rnn.y_hat'][:, 2] - 2.5, 'C3')
    plt.xlim([0, 100])
    plt.yticks([])
    plt.xlabel('time steps')

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




























