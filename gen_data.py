#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 16:30:03 2018

@author: omarschall
"""

import numpy as np

def gen_data(N, n1, n2, one_hot=True):
    
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
            
            y = np.random.binomial(1, p)
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

