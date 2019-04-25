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
from functions import *
from itertools import product
import os
import pickle
from copy import copy
from state_space import State_Space_Analysis
from pdb import set_trace
from scipy.stats import linregress

if os.environ['HOME']=='/home/oem214':
    n_seeds = 10
    try:
        i_job = int(os.environ['SLURM_ARRAY_TASK_ID']) - 1
    except KeyError:
        i_job = 0
    macro_configs = config_generator(algorithm=['DNI', 'RTRL'])
    micro_configs = tuple(product(macro_configs, list(range(n_seeds))))
    
    params, i_seed = micro_configs[i_job]
    i_config = i_job//n_seeds
    np.random.seed(i_seed)
    
    save_dir = os.environ['SAVEPATH']
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

if os.environ['HOME']=='/Users/omarschall':
    params = {'algorithm': 'RTRL'}
    i_job = 0
    save_dir = '/Users/omarschall/vanilla-rtrl/library'

if False:
    data_dir = '/scratch/oem214/vanilla-rtrl/library/ssa_2_run'
    
    i_file = i_job
    file_name = 'rnn_{}'.format(i_file)
    test_file_name = 'rnn_{}_test_data'.format(i_file)
    
    data_path = os.path.join(data_dir, test_file_name)
    rnn_path = os.path.join(data_dir, file_name)
    with open(data_path, 'rb') as f:
        test_data = pickle.load(f)
    with open(rnn_path, 'rb') as f:
        result = pickle.load(f)
        
    n_trials = len(test_data.keys())
    alignments = np.zeros((n_trials, n_trials))
    for i in range(n_trials):
        for j in range(n_trials):
            alignments[i,j] = np.square(test_data[i]['PCs'].T.dot(test_data[j]['PCs'])).sum()/3
    
    result = {}
    result[file_name+'_alignments'] = alignments
    
    if os.environ['HOME']=='/home/oem214':
    
        save_dir = os.environ['SAVEPATH']
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        save_path = os.path.join(save_dir, 'rnn_{}_analysis'.format(i_file))
        
        with open(save_path, 'wb') as f:
            pickle.dump(result, f)

if True:

    #task = Coin_Task(4, 6, one_hot=True, deterministic=True, tau_task=1)
    #time_steps_per_trial = 30
    #task = Sine_Wave(1/time_steps_per_trial, [1, 0.7, 0.3, 0.1], method='regular', never_off=True)
    task = Sensorimotor_Mapping(t_report=7, t_stim=1, stim_duration=3, report_duration=3)
    reset_sigma = 0.05
    data = task.gen_data(50000, 1000)
    
    n_in     = task.n_in
    n_hidden = 20
    n_out    = task.n_out
    
    W_in  = np.random.normal(0, np.sqrt(1/(n_in)), (n_hidden, n_in))
    W_rec = np.linalg.qr(np.random.normal(0, 1, (n_hidden, n_hidden)))[0]
    #W_rec = np.random.normal(0, np.sqrt(1/n_hidden), (n_hidden, n_hidden))
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
    
    optimizer = SGD(lr=0.0003)#, lr_decay_rate=0.999999, min_lr=0.00001)#, clipnorm=5)
    #KeRNL_optimizer = SGD(lr=0.001)
    
    if params['algorithm']=='DNI':
        SG_optimizer = SGD(lr=0.005)
        learn_alg = DNI(rnn, SG_optimizer, W_a_lr=0.01, backprop_weights='exact',
                        SG_label_activation=tanh, W_FB=W_FB)
                        #train_SG_with_exact_CA=False)
    #learn_alg.SG_init(8)
    if params['algorithm']=='RTRL':
        learn_alg = RTRL(rnn)
    if params['algorithm']=='BPTT':                    
        T = task.time_steps_per_trial
        learn_alg = Forward_BPTT(rnn, T)
    #learn_alg = RFLO(rnn, alpha=params['alpha_RFLO'], W_FB=W_FB)
    #learn_alg = KeRNL(rnn, KeRNL_optimizer, sigma_noise=0.01)
    #cCclearn_alg = BPTT(rnn, 1, 10)
    #monitors = ['loss_', 'y_hat', 'sg_loss', 'loss_a', 'sg_target-norm', 'global_grad-norm', 'A-norm', 'a-norm']
    #monitors += ['CA_forward_est', 'CA_SG_est']
    monitors = ['loss_', 'y_hat', 'lr']#, 'sg_loss', 'loss_a', 'sg', 'CA', 'W_rec_alignment']
    
    sim = Simulation(rnn, learn_alg, optimizer, L2_reg=0.0005,
                     time_steps_per_trial=task.time_steps_per_trial,
                     reset_sigma=reset_sigma,
                     reset_at_trial_start=True,
                     i_job=i_job,
                     save_dir=save_dir)
#                     SSA_PCs=3)
    sim.run(data,
            monitors=monitors,
            verbose=True,
            check_accuracy=True)
    
    #sim = Simulation(rnn, learn_alg, optimizer, L2_reg=0.00001,
    #                 time_steps_per_trial=task.time_steps_per_trial,
    #                 reset_sigma=reset_sigma,
    #                 trial_lr_mask=np.sqrt(task.trial_lr_mask))
    #
    #sim.run(data,
    #        monitors=monitors,
    #        verbose=True)
    
    if os.environ['HOME']=='/Users/omarschall':
        
        #Test run
        n_test = 10000
        data = task.gen_data(100, n_test)
        test_sim = Simulation(sim.net, learn_alg=None, optimizer=None,
                              time_steps_per_trial=task.time_steps_per_trial,
                              reset_sigma=reset_sigma,
                              reset_at_trial_start=True)
        test_sim.run(data, mode='test', monitors=['loss_', 'y_hat', 'a', 'a-norm'])
        plt.figure()
        plt.plot(test_sim.mons['y_hat'][:,0])
        plt.plot(data['test']['Y'][:,0])
        plt.plot(data['test']['X'][:,0])
        #plt.plot(test_sim.mons['a-norm'])
        plt.legend(['Prediction', 'Label', 'Stimulus', 'A Norm'])
        #plt.ylim([0, 1.2])
        for i in range(n_test//task.time_steps_per_trial):
            plt.axvline(x=i*task.time_steps_per_trial, color='k', linestyle='--')
        plt.xlim([0, 100])
        
        plt.figure()
        plot_filtered_signals([sim.mons['loss_']], plot_loss_benchmarks=False)
        
    #    plt.figure()
    #    plt.plot(test_sim.mons['y_hat'][:,0],data['test']['Y'][:,0], '.', alpha=0.05)
    #    plt.plot([0, 1], [0, 1], 'k', linestyle='--')
    #    #plt.axis('equal')
    #    plt.ylim([0, 1.1])
    #    plt.xlim([0, 1.1])
    
    if os.environ['HOME']=='/home/oem214':
    
        result = {'sim': sim, 'i_seed': i_seed, 'task': task,
                  'config': params, 'i_config': i_config, 'i_job': i_job}
        save_dir = os.environ['SAVEPATH']
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        save_path = os.path.join(save_dir, 'rnn_'+str(i_job))
        
        with open(save_path, 'wb') as f:
            pickle.dump(result, f)




























