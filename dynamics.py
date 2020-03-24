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

def find_slow_points(args, LR=1e-6, N_iters=100000,
                     return_whole_optimization=True,
                     return_period=5000,
                     stopping_criterion=5):

    test_sim = args[0]
    i_seed_1 = args[1]
    i_seed_2 = args[2]
    np.random.seed(i_seed_1*20 + i_seed_2)
    rnn = test_sim.rnn
    test_data = test_sim.mons['rnn.a']
    n_stop = 0

    a_values = []
    i_a = np.random.randint(test_data.shape[0])
    rnn.reset_network(a=test_data[i_a])
    speeds = [rnn.get_network_speed()]
    norms = [norm(rnn.a)]
    for i_iter in range(N_iters):
        speeds.append(rnn.get_network_speed())
        norms.append(norm(rnn.a))
        rnn.a -= LR * rnn.get_network_speed_gradient()
        a_values.append(np.copy(rnn.a))

        #Stop optimization if speed increases too many steps in a row
        if speeds[i_iter] > speeds[i_iter - 1]:
            n_stop += 1
            if n_stop >= stopping_criterion:
                break
        else:
            n_stop = 0

    if return_whole_optimization:
        return np.array(a_values[::return_period]+[a_values[-1]]), speeds
    else:
        return a_values[-1], speeds