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
    macro_configs = config_generator(algorithm=['E-BPTT', 'RFLO'])
    micro_configs = tuple(product(macro_configs, list(range(n_seeds))))

    params, i_seed = micro_configs[i_job]
    i_config = i_job//n_seeds
    np.random.seed(i_job)

    save_dir = os.environ['SAVEPATH']
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
        
if os.environ['HOME'] == '/Users/omarschall':
    params = {'algorithm': 'E-BPTT'}
    i_job = 0
    save_dir = '/Users/omarschall/vanilla-rtrl/library'

    #np.random.seed(1)

# with open('notebooks/good_ones/sim_006', 'rb') as f:
#     sim = pickle.load(f)


# checkpoint = sim.checkpoints[params['i_checkpoint']]

task = Flip_Flop_Task(3, 0.05, tau_task=1)
#data = task.gen_data(3000000, 10000)
data = task.gen_data(30, 10000)


# result = find_KE_minima(checkpoint, data, N=50, parallelize=True,
#                         N_iters=1000000, verbose=True)

# rnn = deepcopy(checkpoint['rnn'])

# transform = Vanilla_PCA
# #ssa = State_Space_Analysis(checkpoint, data, transform)

# test_sim = Simulation(rnn)
# test_sim.run(data,
#               mode='test',
#               monitors=['rnn.loss_', 'rnn.y_hat', 'rnn.a'],
#               verbose=False)


#ssa.clear_plot()
#ssa.plot_in_state_space(test_sim.mons['rnn.a'][1000:], marker='.', alpha=0.01)
#ssa.plot_in_state_space(results['a_trajectory'], color='C1')

# rnn.reset_network()
# results = find_KE_minimum()

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

optimizer = SGD_Momentum(lr=0.0005, mu=0.6, clip_norm=0.3)
if params['algorithm'] == 'E-BPTT':
    learn_alg = Efficient_BPTT(rnn, 10, L2_reg=0.0001)
elif params['algorithm'] == 'RFLO':
    learn_alg = RFLO(rnn, alpha=alpha, L2_reg=0.0001)
#learn_alg = RFLO(rnn, alpha=alpha)
#learn_alg = Only_Output_Weights(rnn)
#learn_alg = RTRL(rnn, M_decay=0.7)
#learn_alg = RFLO(rnn, alpha=alpha)

comp_algs = []
#monitors = ['learn_alg.rec_grads-norm', 'rnn.loss_']
monitors = []

sim = Simulation(rnn)
sim.run(data, learn_alg=learn_alg, optimizer=optimizer,
        comp_algs=comp_algs,
        monitors=monitors,
        verbose=True,
        report_accuracy=False,
        report_loss=True,
        checkpoint_interval=500000)

# with open('notebooks/good_ones/current_fave', 'wb') as f:
#     pickle.dump(sim, f)

with open('notebooks/good_ones/current_fave', 'rb') as f:
    sim = pickle.load(f)
    rnn = sim.rnn
    
# with open('/Users/omarschall/cluster_results/vanilla-rtrl/rflo_bptt/result_0', 'rb') as f:
#     result = pickle.load(f)
#     sim = result['sim']
#     rnn = sim.rnn

N_iters = 10000
same_LR_criterion = 4000

for i_checkpoint in [1, 3, -1]:
    
    rnn = sim.checkpoints[i_checkpoint]['rnn']
    test_sim = Simulation(rnn)
    test_sim.run(data,
                  mode='test',
                  monitors=['rnn.loss_', 'rnn.y_hat', 'rnn.a'],
                  verbose=False)
    
    transform = Vanilla_PCA
    ssa = State_Space_Analysis(sim.checkpoints[i_checkpoint], data, transform)
    V = ssa.transform(np.eye(n_hidden))

    fixed_points, initial_states = find_KE_minima(sim.checkpoints[i_checkpoint], data, N=300,
                                                  PCs=V, sigma_pert=0, N_iters=N_iters, LR=1,
                                                  weak_input=0, parallelize=True,
                                                  verbose=True, same_LR_criterion=same_LR_criterion)
    
    # with open('notebooks/good_ones/current_fave_FPs', 'wb') as f:
    #     pickle.dump(fixed_points, f)
    
    A = np.array([d['a_final'] for d in fixed_points])
    A_init = np.array(initial_states)
    KE = np.array([d['KE_final'] for d in fixed_points])
    
    dbscan = DBSCAN(eps=0.5)
    dbscan.fit(A)
    dbscan.labels_
    
    A_eigs = []
    for i in range(A.shape[0]):
        
        rnn.reset_network(a=A[i])
        a_J = rnn.get_a_jacobian(update=False)
        A_eigs.append(np.abs(np.linalg.eig(a_J)[0][0]))
    A_eigs = np.array(A_eigs)
    

    ssa.clear_plot()
    ssa.plot_in_state_space(test_sim.mons['rnn.a'][1000:], False, 'C0', '.', alpha=0.05)
    ssa.plot_in_state_space(A[A_eigs>1], False, 'C1', '*', alpha=1)
    ssa.plot_in_state_space(A[A_eigs<1], False, 'C2', '*', alpha=1)
    ssa.plot_in_state_space(A_init, False, 'C3', 'x', alpha=1)
    
    cluster_idx = np.unique(dbscan.labels_)
    n_clusters = len(cluster_idx) - (-1 in cluster_idx)
    cluster_means = np.zeros((n_clusters, n_hidden))
    for i in np.unique(dbscan.labels_):
        
        if i == -1:
            color = 'k'
            continue
        else:
            color = 'C{}'.format(i+1)
            cluster_means[i] = A[dbscan.labels_ == i].mean(0)
            
        
        ssa.plot_in_state_space(A[dbscan.labels_ == i], False, color,
                                '*', alpha=0.5)
    
    ssa.plot_in_state_space(cluster_means, False, 'k', 'X')
    
    cluster_eigs = []
    for i in range(cluster_means.shape[0]):
        
        rnn.reset_network(a=cluster_means[i])
        a_J = rnn.get_a_jacobian(update=False)
        cluster_eigs.append(np.abs(np.linalg.eig(a_J)[0][0]))
    cluster_eigs = np.array(cluster_eigs)
    
    plt.figure()
    plt.hist(cluster_eigs)
    plt.title('Checkpoint {}'.format(i_checkpoint))
    # for i in range(2, 6):
    #     rnn.reset_network(a=test_sim.mons['rnn.a'][i*100])
    #     result = find_KE_minimum(rnn, return_whole_optimization=True, verbose=True,
    #                               N_iters=N_iters, same_LR_criterion=same_LR_criterion)
    #     ssa.plot_in_state_space(result['a_trajectory'], True, 'C{}'.format(i), '-', alpha=1)


    ssa.fig.suptitle('Checkpoint {}'.format(i_checkpoint))



    # fig = plt.figure()
    # plt.plot(test_sim.mons['rnn.y_hat'][:, 0])
    # #plt.plot(data['test']['X'][:, 0])
    # plt.plot(data['test']['Y'][:, 0])
    # plt.xlim([2000, 3000])
    # plt.title('Checkpoint {}'.format(i_checkpoint))
        
    plt.figure()
    plt.hist(np.log10(KE), bins=20, color='C1')
    plt.title('Checkpoint {}'.format(i_checkpoint))
    plt.figure()
    plt.hist(A_eigs, bins=20, color='C2')
    plt.title('Checkpoint {}'.format(i_checkpoint))
#ssa.fig

# rnn_copy = deepcopy(rnn)
# rnn_copy.reset_network(a=A[28]+np.random.normal(0,0.3,64))
# result = find_KE_minimum(rnn_copy, return_whole_optimization=True, verbose=True)

# A_eigs = []
# for i in range(A.shape[0]):
    
#     rnn.reset_network(a=A[i])
#     a_J = rnn.get_a_jacobian(update=False)
#     A_eigs.append(np.abs(np.linalg.eig(a_J)[0][0]))
# A_eigs = np.array(A_eigs)
# test_sim = Simulation(rnn)
# test_sim.run(data,
#              mode='test',
#              monitors=['rnn.loss_'],
#              verbose=False)
# test_loss = np.mean(test_sim.mons['rnn.loss_'])
# processed_data = {'test_loss': test_loss}

#fixed_points = find_KE_minima(sim.checkpoints[-1], data, N=200, parallelize=True,
#                              N_iters=100000, verbose=True)

# test_sim = Simulation(rnn)
# test_sim.run(data,
#               mode='test',
#               monitors=['rnn.loss_', 'rnn.y_hat', 'rnn.a'],
#               verbose=False)

plt.figure()
plt.plot(test_sim.mons['rnn.y_hat'][:, 1])
#plt.plot(data['test']['X'][:, 1])
plt.plot(data['test']['Y'][:, 1])
plt.xlim([0, 1000])

plt.figure()
plt.hist(np.log10(KE), bins=20)

processed_data = {}

if os.environ['HOME'] == '/Users/omarschall' and False:

    transform = partial(UMAP_, min_dist=0, n_neighbors=30)
    ssa = State_Space_Analysis(sim.checkpoints[-1], data, transform)
    ssa.clear_plot()
    ssa.plot_in_state_space(test_sim.mons['rnn.a'], marker='.', alpha=0.01)
    transform = Vanilla_PCA
    ssa_2 = State_Space_Analysis(sim.checkpoints[-1], data, transform)
    ssa_2.clear_plot()
    ssa_2.plot_in_state_space(test_sim.mons['rnn.a'], marker='.', alpha=0.01)
    #ssa.fig.axes[0].set_xlim([-0.6, 0.6])
    #ssa.fig.axes[0].set_ylim([-0.6, 0.6])
    #ssa.fig.axes[0].set_zlim([-0.8, 0.8])
#    for i in range(8):
#        col = 'C{}'.format(i+1)
#        ssa.plot_in_state_space(A[i][:-1,:], color=col)
#        ssa.plot_in_state_space(A[i][-1,:].reshape((1,-1)), 'x', color=col)

if os.environ['HOME'] == '/home/oem214':

    result = {'sim': sim, 'i_seed': i_seed, 'task': task,
              'config': params, 'i_config': i_config, 'i_job': i_job,
              'processed_data': processed_data}
    save_dir = os.environ['SAVEPATH']
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_path = os.path.join(save_dir, 'result_'+str(i_job))

    with open(save_path, 'wb') as f:
        pickle.dump(result, f)




























