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
from dynamics import find_slow_points
import multiprocessing as mp

if os.environ['HOME'] == '/home/oem214':
    n_seeds = 20
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

    np.random.seed(1)
        
# Load network
network_name = 'j_boxman'
with open(os.path.join('notebooks/good_ones', network_name), 'rb') as f:
    rnn = pickle.load(f)

task = Flip_Flop_Task(3, 0.05)
np.random.seed(0)
n_test = 10000
data = task.gen_data(0, n_test)
#test_sim = deepcopy(sim)
test_sim = Simulation(rnn)
test_sim.run(data,
             mode='test',
             monitors=['rnn.loss_', 'rnn.y_hat', 'rnn.a'],
             verbose=False)

pool = mp.Pool(mp.cpu_count())
n_seeds = 8
results = pool.map_async(find_slow_points, zip([test_sim]*n_seeds,
                                               range(n_seeds), [i_job]*n_seeds))
result = results.get()
pool.close()
A = np.array([result[i][0] for i in range(n_seeds)])
speeds = np.array([result[i][1] for i in range(n_seeds)])

result = {'A': A, 'speeds': speeds}



if os.environ['HOME'] == '/Users/omarschall':

    ssa = State_Space_Analysis(test_sim.mons['rnn.a'], n_PCs=3)
    ssa.plot_in_state_space(test_sim.mons['rnn.a'], '.', alpha=0.002)
    ssa.fig.axes[0].set_xlim([-0.6, 0.6])
    ssa.fig.axes[0].set_ylim([-0.6, 0.6])
    ssa.fig.axes[0].set_zlim([-0.8, 0.8])
    for i in range(8):
        col = 'C{}'.format(i+1)
        ssa.plot_in_state_space(A[i,:-1,:], color=col)
        ssa.plot_in_state_space(A[i,-1,:].reshape((1,-1)), 'x', color=col)

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




























