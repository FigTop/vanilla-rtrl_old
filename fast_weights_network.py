#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 15:05:51 2018

@author: omarschall
"""

import numpy as np
from utils import *
from optimizers import *
from analysis_funcs import *
import time
from copy import copy
from pdb import set_trace
from network import RNN

class Fast_Weights_RNN(RNN):
    
    def __init__(self, W_in, W_rec, W_out, b_rec, b_out,
                 activation, alpha, output, loss, A, n_S, eta, lmbda):
        
        super().__init__(W_in, W_rec, W_out, b_rec, b_out,
                       activation, alpha, output, loss)
        
        self.A = A
        self.n_S = n_S
        self.eta = eta
        self.lmbda = lmbda
    
    def next_state(self, x, a=None, update=True):
        '''
        Accepts as argument the current time step's input x and updates
        the state of the RNN, while storing the previous state h
        and activatation a.
        '''
        
        self.h_prev = np.copy(self.h)
        self.a_prev = np.copy(self.a)
        
        self.h_0 = self.W_rec.dot(self.a) + self.W_in.dot(self.x) + self.b_rec
        self.a_0 = self.activation.f(self.h_0)
        
        self.a_s = np.zeros((self.n_S, self.n_hidden))
        self.a_s[0,:] = self.a_0
        
        for s in range(1,self.n_S):
            
            self.h_s = layer_normalization.f(self.h_0 + self.A.dot(self.a_s[s-1,:]))
            self.a_s[s,:] = self.activation.f(self.h_s)
        
        if update:
            self.h = np.copy(self.h_s)
            self.a = np.copy(self.a_s[-1,:])
        else:
            h = np.copy(self.h_s)
            return (1 - self.alpha)*a + self.activation.f(h)
        
    def update_A(self, a_mean=0):
        
        self.A = self.lmbda*self.A + self.eta*np.multiply.outer(self.a - a_mean, self.a - a_mean)