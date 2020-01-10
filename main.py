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
from metalearning_algorithms import *
from functions import *
from itertools import product
import os
import pickle
from copy import copy
from state_space import State_Space_Analysis
from pdb import set_trace
from scipy.stats import linregress
from scipy.ndimage.filters import uniform_filter1d

if os.environ['HOME']=='/home/oem214':
    n_seeds = 20
    try:
        i_job = int(os.environ['SLURM_ARRAY_TASK_ID']) - 1
    except KeyError:
        i_job = 0
    macro_configs = config_generator(algorithm=['Only_Output_Weights',
                                                'RTRL', 'UORO', 'KF-RTRL', 'R-KF-RTRL',
                                                'BPTT', 'DNI', 'DNIb',
                                                'RFLO', 'KeRNL'],
                                     alpha=[1, 0.5],
                                     task=['Coin', 'Mimic'])
    micro_configs = tuple(product(macro_configs, list(range(n_seeds))))

    params, i_seed = micro_configs[i_job]
    i_config = i_job//n_seeds
    np.random.seed(i_job)

    save_dir = os.environ['SAVEPATH']
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

if os.environ['HOME']=='/Users/omarschall':
    params = {'algorithm': 'RTRL'}
    i_job = 0
    save_dir = '/Users/omarschall/vanilla-rtrl/library'

    np.random.seed()


#task = Sensorimotor_Mapping(t_report=15, report_duration=4)
task = Add_Task(6, 10, deterministic=True, tau_task=1)
task.time_steps_per_trial = 60
task.trial_mask = np.ones(task.time_steps_per_trial)
#task = Sine_Wave(p_transition=0.05, frequencies=[0.03, 0.1, 0.01], method='regular',
#                 never_off=False)

data = task.gen_data(80000, 1000)

n_in     = task.n_in
n_hidden = 32
n_out    = task.n_out

W_in  = np.random.normal(0, np.sqrt(1/(n_in)), (n_hidden, n_in))
W_rec = np.linalg.qr(np.random.normal(0, 1, (n_hidden, n_hidden)))[0]
#W_rec = np.random.normal(0, np.sqrt(1/n_hidden), (n_hidden, n_hidden))
#W_rec = (W_rec + W_rec.T)/2
#W_rec = 0.54*np.eye(n_hidden)
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

optimizer = SGD(lr=0.0005)
SG_optimizer = SGD(lr=0.005)


if params['algorithm'] == 'Only_Output_Weights':
    learn_alg = Only_Output_Weights(rnn)
if params['algorithm'] == 'RTRL':
    learn_alg = RTRL(rnn)
if params['algorithm'] == 'BPTT':
    learn_alg = Forward_BPTT(rnn, 20)
if params['algorithm'] == 'DNI':
    learn_alg = DNI(rnn, SG_optimizer)
if params['algorithm'] == 'DNIb':
    W_a_lr = 0.001
    learn_alg = DNI(rnn, SG_optimizer, use_approx_J=True, W_a_lr=W_a_lr,
                    SG_label_activation=tanh, W_FB=W_FB)
    learn_alg.name = 'DNIb'
if params['algorithm'] == 'RFLO':
    learn_alg = RFLO(rnn, alpha=alpha)


learn_alg = Reward_Modulated_Hebbian_Plasticity(rnn, alpha=alpha, task=task,
                                                fixed_modulation=None)
learn_alg = UORO(rnn)
#learn_alg = Only_Output_Weights(rnn)
J_lr = 0.005
learn_alg = DNI(rnn, SG_optimizer, use_approx_J=True, J_lr=J_lr,
                SG_label_activation=tanh, W_FB=W_FB)
#learn_alg.name = 'DNIb'
#learn_alg = COLIN(rnn, decay=0.2, sigma=0.1, task=task)
#learn_alg = Only_Output_Weights(rnn)
comp_algs = []

A = np.random.normal(0, 10, n_hidden)
B = np.random.normal(0, 10, (n_hidden, n_hidden + n_in + 1))
learn_alg = UORO(rnn, A=A, B=B)

monitors = ['net.loss_', 'net.y_hat', 'optimizer.lr',
            'learn_alg.running_loss_avg', 'net.W_rec',
            'learn_alg.modulation']
monitors = ['net.loss_', 'net.y_hat', 'learn_alg.A-norm', 'learn_alg.B-norm']

sim = Simulation(rnn,
                 time_steps_per_trial=task.time_steps_per_trial,
                 reset_sigma=None,
                 trial_mask=task.trial_mask)
sim.run(data, learn_alg=learn_alg, optimizer=optimizer,
        comp_algs=comp_algs,
        monitors=monitors,
        verbose=True,
        check_accuracy=False,
        check_loss=True,
        sigma=0.03)

if os.environ['HOME']=='/Users/omarschall':


    plot_filtered_signals([sim.mons['net.loss_'],
                           sim.mons['learn_alg.A-norm'],
                           sim.mons['learn_alg.B-norm']])

    #Test run
    np.random.seed(2)
    n_test = 1000
    data = task.gen_data(100, n_test)
    test_sim = copy(sim)
    test_sim.run(data,
                 mode='test',
                 monitors=['net.loss_', 'net.y_hat', 'net.a'],
                 verbose=False)
    fig = plt.figure()
    plt.plot(test_sim.mons['net.y_hat'][:,0])
    plt.plot(data['test']['Y'][:,0])
    plt.plot(data['test']['X'][:,0]*0.1)
    #plt.legend(['Prediction', 'Label', 'Stimulus'])#, 'A Norm'])
    #plt.ylim([-0.2, 0.2])
    for i in range(n_test//task.time_steps_per_trial):
        continue
        plt.axvline(x=i*task.time_steps_per_trial, color='k', linestyle='--')
    plt.xlim([0, 200])
    #fig.savefig()

    plt.figure()
    x = test_sim.mons['net.y_hat'].flatten()
    y = data['test']['Y'].flatten()
    plt.plot(x, y, '.', alpha=0.05)
    plt.plot([np.amin(x), np.amax(x)],
              [np.amin(y), np.amax(y)], 'k', linestyle='--')
    plt.axis('equal')

if os.environ['HOME']=='/home/oem214':

    result = {'sim': sim, 'i_seed': i_seed, 'task': task,
              'config': params, 'i_config': i_config, 'i_job': i_job,
              'processed_data': processed_data}
    save_dir = os.environ['SAVEPATH']
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_path = os.path.join(save_dir, 'rnn_'+str(i_job))

    with open(save_path, 'wb') as f:
        pickle.dump(result, f)




























