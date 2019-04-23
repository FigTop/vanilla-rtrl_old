#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  3 12:08:30 2019

@author: omarschall
"""

import numpy as np
import itertools

### --- Mathematical tools --- ###

def norm(z):

   return np.sqrt(np.sum(np.square(z)))

def split_weight_matrix(A, sizes, axis=1):

    indices = [0] + np.cumsum(sizes).tolist()
    return [np.squeeze(A[:,indices[i]:indices[i+1]]) for i in range(len(indices) - 1)]

def rectangular_filter(signal, filter_size=100):

    return np.convolve(signal, np.ones(filter_size)/filter_size, mode='valid')

def classification_accuracy(data, y_hat):

    y_hat = np.array(y_hat)

    i_label = np.argmax(data['test']['Y'], axis=1)
    i_pred = np.argmax(y_hat, axis=1)

    acc = np.sum(i_label==i_pred)/len(i_label)

    return acc

def normalized_dot_product(a, b):
    
    a_norm = norm(a)
    b_norm = norm(b)
    
    if a_norm >0 and b_norm > 0:
        return np.dot(a.flatten(),b.flatten())/(a_norm*b_norm)
    else:
        return 0

def get_spectral_radius(M):

    eigs, _ = np.linalg.eig(M)

    return np.amax(np.absolute(eigs))

### --- Programming tools --- ###

def config_generator(**kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))