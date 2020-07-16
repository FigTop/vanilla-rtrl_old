#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 16:30:58 2018
@author: omarschall
"""
#%%
import numpy as np
from network import RNN
from lstm_network import LSTM
from lstm_learning_algorithms import *
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

print('Home env: ',os.environ['HOME'] )

if os.environ['HOME'] == '/home/yx2105':
    n_seeds = 1
    try:
        i_job = int(os.environ['SLURM_ARRAY_TASK_ID']) - 1
    except KeyError:
        i_job = 0
    macro_configs = config_generator(units = [16,32,64], lr = [0.0001],optimizer=['Adam'],algorithm = ['uoro'], beta=[0.9],T=[20])
    micro_configs = tuple(product(macro_configs, list(range(n_seeds))))

    params, i_seed = micro_configs[i_job]
    i_config = i_job//n_seeds
    np.random.seed(0)

    save_dir = os.environ['SAVEPATH']
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

if os.environ['HOME'] == '/Users/yanqixu':
    params = {'optimizer':'SGD', 'algorithm':'F-BPTT', 'lr':0.1, 'units':64,'T':50,'beta':0.9}
    i_job = 0
    save_dir = '/Users/yanqixu/Documents/1.0.MasterCDS/OnlineRNN/vanilla-rtrl/library'
    np.random.seed(0)

task = Add_Task(3, 5, deterministic=True, tau_task=1)
data = task.gen_data(50000, 10000) #10000000

n_in = task.n_in
n_h = params['units']
n_out = task.n_out
n_h_hat = n_h + n_in
n_t = 2 * n_h

W_f  = np.random.normal(0, np.sqrt(1/(n_h_hat)), (n_h, n_h_hat))
W_i  = np.random.normal(0, np.sqrt(1/(n_h_hat)), (n_h, n_h_hat))
W_a  = np.random.normal(0, np.sqrt(1/(n_h_hat)), (n_h, n_h_hat))
W_o  = np.random.normal(0, np.sqrt(1/(n_h_hat)), (n_h, n_h_hat))
b_f = np.zeros(n_h)
b_i = np.zeros(n_h)
b_a = np.zeros(n_h)
b_o = np.zeros(n_h)
W_out = np.random.normal(0, np.sqrt(1/(n_t)), (n_out, n_t))
b_out = np.zeros(n_out)


lstm = LSTM(W_f, W_i, W_a, W_o, W_out, 
            b_f, b_i, b_a, b_o, b_out,
            output=softmax,
            loss=softmax_cross_entropy)

if params['optimizer'] == 'SGD':
    optimizer = Stochastic_Gradient_Descent(params['lr'])
elif params['optimizer'] == 'SGDmom':
    optimizer = SGD_Momentum(params['lr'], mu=0.01, clip_norm=0.3) #mu=0.01
elif params['optimizer'] == 'Adam':
    optimizer = Adam(lr = params['lr'],beta_1=params['beta'])


if params['algorithm'] == 'output':
    learn_alg = Only_Output_LSTM(lstm)
elif params['algorithm'] == 'uoro':
    learn_alg = UORO_LSTM(lstm,nu_dist='gaussian')
elif params['algorithm'] == 'rtrl':
    learn_alg = RTRL(lstm) #0.1
elif params['algorithm'] == 'kf':
    learn_alg = KF_RTRL_LSTM(lstm)
elif params['algorithm'] == 'E-BPTT':
    learn_alg = Efficient_BPTT_LSTM(lstm, T_truncation=params['T'])
elif params['algorithm'] == 'F-BPTT':
    learn_alg = Future_BPTT_LSTM(lstm, T_truncation=params['T'])


# run algorithm
comp_algs = []
monitors = ['rnn.loss_']

sim = Simulation(lstm)
sim.run(data, learn_alg=learn_alg, optimizer=optimizer,
        comp_algs=comp_algs,
        monitors=monitors,
        verbose=True,
        report_accuracy=False,
        report_loss=True,
        checkpoint_interval=None)




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

if os.environ['HOME'] == '/home/yx2105':

    result = {'sim': sim, 'i_seed': i_seed, 'task': task,
              'config': params, 'i_config': i_config, 'i_job': i_job,
              'processed_data': None}
    result['average_loss'] = sim.avg_loss_list

    save_dir = os.environ['SAVEPATH']
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_path = os.path.join(save_dir, 'result_'+str(i_job))

    with open(save_path, 'wb') as f:
        pickle.dump(result, f)




#%%
import matplotlib.pyplot as plt
#sim.mons
plt.plot(sim.mons['rnn.f'][:20])

#np.save('UORO_papw',sim.mons['rnn.papw'])
#np.save('UORO_papwc',sim.mons['rnn.papw_c'])
#plt.spy(sim.mons['rnn.papwf'][2000][:,-96:])
# papwf = sim.mons['rnn.papw'][0][:,:1120]
# papwf
# plt.spy(sim.mons['rnn.papw'][0][:,:1120])
# plt.spy(sim.mons['rnn.papw'][0][:,1120:2240]
# plt.spy(sim.mons['rnn.papw'][0][:,2240:3360])
# plt.spy(sim.mons['rnn.papw'][0][:,3360:])
#sim.mons['rnn.papwf'][19999]









# %%
