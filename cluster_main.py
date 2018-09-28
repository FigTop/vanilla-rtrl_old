#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 17:54:50 2018

@author: omarschall
"""

import numpy as np
from network import RNN
from utils import *
from gen_data import gen_data
from optimizers import *
import warnings
import os
import pickle

try:
    i_job = int(os.environ['SLURM_ARRAY_TASK_ID']) - 1
except KeyError:
    i_job = 0

n_tries = 100
T = 25000

n_hidden = 32
n_back = [3, 5, 8]
methods = ['rtrl', 'kf']
method = methods[i_job]
seeds = list(range(n_tries))

data = {}
for n1 in n_back:
    for seed in seeds:
        
        key = str(n_hidden)+'_'+str(n1)+'_'+method+'_'+'seed='+str(seed)
        np.random.seed(seed)

        n_in     = 2
        n_out    = 2
                            
        W_in  = np.random.normal(0, np.sqrt(1/(n_in + n_hidden)), (n_hidden, n_in))
        W_rec = np.random.normal(0, np.sqrt(1/(n_hidden + n_hidden)), (n_hidden, n_hidden))
        W_out = np.random.normal(0, np.sqrt(1/(n_hidden + n_hidden)), (n_out, n_hidden))
            
        b_rec = np.zeros(n_hidden)
        b_out = np.zeros(n_out)
            
        optimizer = Adam(lr=0.001)
        X, Y = gen_data(T, n1, n1+2, one_hot=True, deterministic=True)
        rnn = RNN(W_in, W_rec, W_out, b_rec, b_out,
                  activation=relu, output=softmax,
                  loss=softmax_cross_entropy)

        with warnings.catch_warnings():
            warnings.simplefilter('error')                
            try:
                losses, y_hats = rnn.run(X, Y, optimizer, method=method)
                data[key] = np.array(losses)
            except RuntimeWarning:
                pass

data['RNN_type_example'] = rnn

np.random.seed()

with open('library/'+method+'_'+str(np.random.rand())[2:6], 'wb') as f:
    pickle.dump(data, f)



