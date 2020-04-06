#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 13:48:33 2020

@author: omarschall
"""

import numpy as np
from simulation import *
from utils import norm
import multiprocessing as mp
from submit_jobs import write_job_file
from functools import partial
from pdb import set_trace
from network import RNN
from copy import copy, deepcopy
import time
import os
import umap

### --- WRAPPER METHODS --- ###

def get_test_sim_data(checkpoint, test_data):
    """Get hidden states from a test run for a given checkpoint"""

    rnn = deepcopy(checkpoint['rnn'])
    test_sim = Simulation(rnn)
    test_sim.run(test_data, mode='test',
                 monitors=['rnn.a'],
                 verbose=False,
                 a_initial=rnn.a.copy())

    return test_sim.mons['rnn.a']

def analyze_all_checkpoints(checkpoints, func, test_data, **kwargs):
    """For a given analysis function and a list of checkpoints, applies
    the function to each checkpoint in the list and returns all results.
    Uses multiprocessing to apply the analysis independently."""

    func_ = partial(func, test_data=test_data, **kwargs)
    pool = mp.Pool(mp.cpu_count())
    results = pool.map(func_, checkpoints)
    pool.close()

    return results

### --- ANALYSIS METHODS --- ###

def Vanilla_PCA(checkpoint, test_data, n_PCs=3):
    """Return first n_PCs PC axes of the test """

    test_a = get_test_sim_data(checkpoint, test_data)
    U, S, V = np.linalg.svd(test_a)

    return V[:,:n_PCs]

def UMAP(checkpoint, test_data, n_components=3, **kwargs):
    """Performs  UMAP with default parameters and returns component axes."""
    
    test_a = get_test_sim_data(checkpoint, test_data)
    fit = umap.UMAP(n_components=n_components, **kwargs)
    u = fit.fit_transform(test_a)
    
    return u

def find_KE_minima(checkpoint, test_data, N=1000, verbose_=False,
                   parallelize=False, **kwargs):
    """Find many KE minima for a given checkpoint of training. Includes option
    to parallelize or not."""

    test_a = get_test_sim_data(checkpoint, test_data)
    results = []

    if parallelize:

        RNNs = []
        for i in range(N):
            rnn = deepcopy(checkpoint['rnn'])
            i_a = np.random.randint(test_a.shape[0])
            a_init = test_a[i_a]
            rnn.reset_network(a=a_init)
            RNNs.append(rnn)

        func_ = partial(find_KE_minimum, **kwargs)
        pool = mp.Pool(mp.cpu_count())
        results = pool.map(func_, RNNs)
        pool.close()

    if not parallelize:
        for i in range(N):

            if i % (N // 10) == 0 and verbose_:
                print('{}% done'.format(i * 10 / N))

            #Pick random test state as starting point
            i_a = np.random.randint(test_a.shape[0])
            a_init = test_a[i_a]

            #Start new network object and reset state to starting point
            rnn = deepcopy(checkpoint['rnn'])
            rnn.reset_network(a=a_init)

            #Calculate final result
            result = find_KE_minimum(rnn, **kwargs)
            results.append(result)

    return results

def find_KE_minimum(rnn, LR=1e-3, N_iters=1000000,
                    return_whole_optimization=False,
                    return_period=100,
                    N_KE_increase=3,
                    LR_drop_factor=5,
                    LR_drop_criterion=12,
                    same_LR_criterion=np.inf,
                    verbose=False,
                    calculate_linearization=False):
    """For a given RNN, performs gradient descent with adaptive learning rate
    to find a kinetic energy  minimum of the network. The seed state is just
    the state of the rnn, rnn.a.


    Returns either ust the final rnn state and KE, or if
    return_whole_optimization is True, then it also returns trajectory of a
    values, norms, and LR_drop times at a frequency specified by
    return_period."""

    #Initialize counters
    i_LR_drop = 0
    i_KE_increase = 0
    i_same_LR = 0

    #Initialize return lists
    a_values = []
    KEs = [rnn.get_network_speed()]
    norms = [norm(rnn.a)]
    LR_drop_times = []

    #Loop once for each iteration
    for i_iter in range(N_iters):

        #Report progress
        if i_iter % (N_iters//10) == 0:
            pct_complete = np.round(i_iter/N_iters*100, 2)
            if verbose:
                print('{}% done'.format(pct_complete))

        rnn.a -= LR * rnn.get_network_speed_gradient()
        a_values.append(rnn.a.copy())

        KEs.append(rnn.get_network_speed())
        norms.append(norm(rnn.a))

        #Stop optimization if KE increases too many steps in a row
        if KEs[i_iter] > KEs[i_iter - 1]:
            i_KE_increase += 1
            if i_KE_increase >= N_KE_increase:
                LR /= LR_drop_factor
                if verbose:
                    print('LR drop #{} at iter {}'.format(i_LR_drop, i_iter))
                LR_drop_times.append(i_iter)
                i_LR_drop += 1
                i_KE_increase = 0
                if i_LR_drop >= LR_drop_criterion:
                    if verbose:
                        print('Reached criterion at {} iter'.format(i_iter))
                    break
        else:
            i_KE_increase = 0
            i_same_LR += 1

        if i_same_LR >= same_LR_criterion:
            print('Reached same LR criterion at {} iter'.format(i_iter))
            break


    results = {'a_final': a_values[-1],
               'KE_final': KEs[-1]}

    if calculate_linearization:

        rnn.a = results['a_final']
        a_J = rnn.get_a_jacobian(update=False)
        eigs, _ = np.linalg.eig(a_J)
        results['jacobian_eigs'] = eigs

    if return_whole_optimization:
        results['a_trajectory'] = np.array(a_values[::return_period])
        results['norms'] = np.array(norms[::return_period])
        results['KEs'] = np.array(KEs[::return_period])
        results['LR_drop_times'] = LR_drop_times

    return results

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

