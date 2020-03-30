#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 13:48:33 2020

@author: omarschall
"""

import numpy as np
from simulation import *
from utils import norm
from pdb import set_trace
from network import RNN
from copy import copy, deepcopy
import time

class Dynamics_Analysis:
    
    def __init__(self, run_period,
                 save_rnn=True, save_learn_alg=True,
                 save_optimizer=True, save_index=True,
                 allowed_kwargs_=set(), **kwargs):
        
        allowed_kwargs = set().union(allowed_kwargs_)

        for k in kwargs:
            if k not in allowed_kwargs:
                raise TypeError('Unexpected keyword argument passed'
                                'to Dynamics_Analysis.__init__: ' + str(k))

        #Set all non-specified kwargs to None
        for attr in allowed_kwargs:
            if not hasattr(self, attr):
                setattr(self, attr, None)

        #Make kwargs attributes of the instance
        self.__dict__.update(kwargs)
        
        self.run_period = run_period
        
        self.results = {'analysis': []}
        if save_rnn:
            self.results['checkpoint_rnn'] = []
        if save_learn_alg:
            self.results['checkpoint_learn_alg'] = []
        if save_optimizer:
            self.results['checkpoint_optimizer'] = []
        if save_index:
            self.results['checkpoint_i_t'] = []
        
    def __call__(self, sim, data):
        
        #Save data for checkpoints
        for key in self.results.keys():
            if key != 'analysis':
                sim_key = key.split('checkpoint_')[-1]
                self.results[key].append(deepcopy(getattr(sim, sim_key)))
        
        #Must have "analysis result" method
        analysis_result = self.run_analysis(sim, data)
        self.results['analysis'].append(analysis_result)


class Vanilla_PCA(Dynamics_Analysis):
    
    def __init__(self, run_period, n_PCs):
        
        allowed_kwargs_ = set()
        super().__init__(run_period, allowed_kwargs_)
        
        self.n_PCs = n_PCs
        
    def run_analysis(self, sim, data):
        
        test_sim = Simulation(deepcopy(sim.rnn))
        test_sim.run(data, mode='test',
                     monitors=['rnn.a'],
                     verbose=False,
                     a_initial=sim.rnn.a.copy())
        
        U, S, V = np.linalg.svd(test_sim.mons['rnn.a'])
        return V[:,:self.n_PCs]
    
class Empty_Analysis(Dynamics_Analysis):
    
    def run_analysis(self, sim, data):
    
        return None

def run_autonomous_sim(rnn, N, a_initial, monitors=[]):
    """Creates and runs a test simulation with no inputs and a specified
    initial state of the network."""

    #Create empty data array
    data = {'test': {'X': np.zeros((N, rnn.n_in)),
                     'Y': np.zeros((N, rnn.n_out))}}
    sim = Simulation(rnn)
    sim.run(data, mode='test', monitors=monitors,
            a_initial=a_initial,
            verbose=True,
            check_accuracy=False,
            check_loss=False)

    return sim

def find_slow_points(args, LR=1e-3, N_iters=100000,
                     N_seed_1=20, N_seed_2=30,
                     return_whole_optimization=True,
                     return_period=5000,
                     stopping_criterion=12,
                     LR_drop_factor=10,
                     LR_criterion=10,
                     same_LR_criterion=10000):

    test_sim = deepcopy(args[0])
    i_seed_1 = args[1]
    i_seed_2 = args[2]
    np.random.seed(i_seed_1*N_seed_2 + i_seed_2)
    rnn = test_sim.rnn
    test_data = test_sim.mons['rnn.a']
    n_stop = 0
    i_LR = LR_criterion - 1
    i_same_LR = 0
    
    a_values = []
    i_a = np.random.randint(test_data.shape[0])
    rnn.reset_network(a=test_data[i_a])
    speeds = [rnn.get_network_speed()]
    norms = [norm(rnn.a)]
    LR_drop_times = []
    for i_iter in range(N_iters):
        
        if i_iter % (N_iters//10) == 0:
            pct_complete = np.round(i_iter/N_iters*100, 2)
            print('Case ({}, {}) {}% done'.format(i_seed_1,
                                                  i_seed_2,
                                                  pct_complete))
            
        rnn.a -= LR * rnn.get_network_speed_gradient()
        a_values.append(np.copy(rnn.a))

        speeds.append(rnn.get_network_speed())
        norms.append(norm(rnn.a))

        #Stop optimization if speed increases too many steps in a row
        if speeds[i_iter] > speeds[i_iter - 1]:
            i_LR += 1
            if i_LR >= LR_criterion:
                LR /= LR_drop_factor
                print('Case ({}, {}) dropping LR'.format(i_seed_1,
                                                         i_seed_2))
                LR_drop_times.append(i_iter)
                n_stop += 1
                if n_stop >= stopping_criterion:
                    print('Case ({}, {}) reached criterion'.format(i_seed_1,
                                                                   i_seed_2))
                    break
        else:
            i_LR = 0
            i_same_LR += 1
            
        if i_same_LR >= same_LR_criterion:
            break
            

    if return_whole_optimization:
        return np.array(a_values[::return_period]+[a_values[-1]]), speeds[::return_period], LR_drop_times
    else:
        return a_values[-1]