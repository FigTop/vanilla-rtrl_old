#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 16:30:03 2018

@author: omarschall
"""

import numpy as np

def coin_task(N, n1, n2, one_hot=True, deterministic=False):
    
    if one_hot:
        
        X = []
        Y = []
        
        for i in range(N):
            
            x = np.random.binomial(1, 0.5)
            X.append(np.array([x, 1-x]))
            
            p = 0.5
            try:
                p += X[-n1][0]*0.5
            except IndexError:
                pass
            try:
                p -= X[-n2][0]*0.25
            except IndexError:
                pass
            
            if not deterministic:
                y = np.random.binomial(1, p)
            else:
                y = p
            Y.append(np.array([y, 1-y]))
        
    else:
        
        X = np.random.binomial(1, 0.5, N).reshape((-1, 1))
        Y = np.zeros_like(X)
        for i in range(N):
            p = 0.5
            if X[i-n1,0]==1:
                p += 0.5
            if X[i-n2,0]==1:
                p -= 0.25
            Y[i,0] = np.random.binomial(1, p)
        
    return X, Y

def copy_task(n_symbols, n_sequences, T):
    
    I = np.eye(n_symbols)
    
    X = np.zeros((1, n_symbols))
    Y = np.zeros((1, n_symbols))
    
    for i in range(n_sequences):
        
        seq = I[np.random.randint(0, n_symbols, size=T)]
        cue = np.tile(np.ones(n_symbols), (T, 1))
        X = np.concatenate([X, seq, cue])
        Y = np.concatenate([Y, np.ones((T, n_symbols))*0.1, seq])
        
    return X, Y

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        