#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 16:30:58 2018

@author: omarschall
"""

import numpy as np
from network import RNN
from lstm_network import LSTM
from lstm_learning_algorithms import Only_Output_LSTM, UORO_LSTM
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
    macro_configs = config_generator()
    micro_configs = tuple(product(macro_configs, list(range(n_seeds))))

    params, i_seed = micro_configs[i_job]
    i_config = i_job//n_seeds
    np.random.seed(i_job)

    save_dir = os.environ['SAVEPATH']
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

if os.environ['HOME'] == '/Users/yanqixu':
    params = {}
    i_job = 0
    save_dir = '/Users/yanqixu//Documents/1.0.MasterCDS/OnlineRNN/vanilla-rtrl/library'
    np.random.seed(0)

task = Add_Task(20, 50, deterministic=True, tau_task=1)
data = task.gen_data(800000, 10000)

# n_in = task.n_in
# n_h = 32
# n_out = task.n_out
# n_h_hat = n_h + n_in
# n_t = 2 * n_h

# W_f  = np.random.normal(0, np.sqrt(1/(n_h_hat)), (n_h, n_h_hat))
# W_i  = np.random.normal(0, np.sqrt(1/(n_h_hat)), (n_h, n_h_hat))
# W_a  = np.random.normal(0, np.sqrt(1/(n_h_hat)), (n_h, n_h_hat))
# W_o  = np.random.normal(0, np.sqrt(1/(n_h_hat)), (n_h, n_h_hat))
# b_f = np.zeros(n_h)
# b_i = np.zeros(n_h)
# b_a = np.zeros(n_h)
# b_o = np.zeros(n_h)
# W_c_out = np.random.normal(0, np.sqrt(1/(n_h)), (n_out, n_h))
# W_h_out = np.random.normal(0, np.sqrt(1/(n_h)), (n_out, n_h))
# b_out = np.zeros(n_out)

# lstm = LSTM(W_f, W_i, W_a, W_o, W_c_out, W_h_out,
#             b_f, b_i, b_a, b_o, b_out,
#             output=softmax,
#             loss=softmax_cross_entropy)

# optimizer = Stochastic_Gradient_Descent(lr=0.001)
# learn_alg = UORO_LSTM(lstm)
# comp_algs = []
# monitors = ['rnn.loss_', 'rnn.y_hat','rnn.h', 'rnn.c','grads_list','rnn.f','rnn.i','rnn.a','rnn.o']

# sim = Simulation(lstm)
# sim.run(data, learn_alg=learn_alg, optimizer=optimizer,
#         comp_algs=comp_algs,
#         monitors=monitors,
#         verbose=True,
#         check_accuracy=False,
#         check_loss=True)


n_in = task.n_in
n_hidden = 32
n_out = task.n_out

W_in  = np.random.normal(0, np.sqrt(1/(n_in)), (n_hidden, n_in))
W_rec = np.concatenate([np.linalg.qr(np.random.normal(0, 1, (n_hidden, n_hidden)))[0],
        W_in],axis=1)
W_out = np.random.normal(0, np.sqrt(1/(n_hidden)), (n_out, n_hidden))
W_FB = np.random.normal(0, np.sqrt(1/n_out), (n_out, n_hidden))

b_rec = np.zeros(n_hidden)
b_out = np.zeros(n_out)

alpha = 1

rnn = RNN(W_rec, W_out, b_rec, b_out,
          activation=tanh,
          alpha=alpha,
          output=softmax,
          loss=softmax_cross_entropy)

optimizer =  Stochastic_Gradient_Descent(lr=0.01)
learn_alg = Only_Output_Weights(rnn)
#learn_alg = UORO(rnn)
#learn_alg = Efficient_BPTT(rnn, T_truncation=100)

comp_algs = []
monitors = ['rnn.loss_']

sim = Simulation(rnn)
sim.run(data, learn_alg=learn_alg, optimizer=optimizer,
        comp_algs=comp_algs,
        monitors=monitors,
        verbose=True,
        report_accuracy=False,
        report_loss=True,
        checkpoint_interval=None)







#Filter losses
#loss = sim.mons['net.loss_']
#downsampled_loss = np.nanmean(loss.reshape((-1, 10000)), axis=1)
#filtered_loss = uniform_filter1d(downsampled_loss, 10)
#processed_data = {'filtered_loss': filtered_loss}
#del(sim.mons['net.loss_'] )

#Get validation losses
np.random.seed(1)
n_test = 10000
data = task.gen_data(0, n_test)
test_sim = deepcopy(sim)
test_sim.run(data,
             mode='test',
             monitors=['rnn.loss_'],
             verbose=False)
test_loss = np.mean(test_sim.mons['rnn.loss_'])
processed_data = {'test_loss': test_loss}

if os.environ['HOME'] == '/Users/omarschall':

    #plot_filtered_signals([sim.mons['net.loss_']])

    #Test run
    np.random.seed(2)
    n_test = 10000
    data = task.gen_data(100, n_test)
    test_sim = deepcopy(sim)
    test_sim.run(data,
                 mode='test',
                 monitors=['rnn.loss_', 'rnn.y_hat', 'rnn.a'],
                 verbose=False)
    fig = plt.figure()
    plt.plot(data['test']['X'][:100,0])
    plt.plot(data['test']['Y'][:100,0])
    for i in range(2):
        plt.plot(test_sim.mons['rnn.y_hat'][:,i], color='C{}'.format(i))
        plt.plot(data['test']['Y'][:,i], color='C{}'.format(i), linestyle='--')
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




























