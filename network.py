#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  7 17:20:39 2018

@author: omarschall
"""

import numpy as np
from utils import *
from functions import *
from optimizers import *
from analysis_funcs import *
import time
from copy import copy
from pdb import set_trace

class RNN:
    
    def __init__(self, W_in, W_rec, W_out, b_rec, b_out, activation, alpha, output, loss):
        '''
        Initializes a vanilla RNN object that follows the forward equation
        
        h_t = (1 - alpha)*h_{t-1} + W_rec * phi(h_{t-1}) + W_in * x_t + b_rec
        z_t = W_out * a_t + b_out
        
        with initial parameter values given by W_in, W_rec, W_out, b_rec, b_in
        and specified activation and loss functions, which must be function
        objects--see utils.py.
        
        ___Arguments___
        
        W_*                 Initial values of (in)put, (rec)urrent and (out)put
                            weights in the network.
                            
        b_*                 Initial values of (rec)urrent and (out)put biases.
        
        activation          Instance of function class (see utils.py) used for
                            calculating activations a from pre-activations h.
                            
        alpha               Ratio of time constant of integration to time constant
                            of leak.
                            
        output              Instance of function class used for calculating final
                            output from z.
                            
        loss                Instance of function class used for calculating loss
                            from z (must implicitly include output function, e.g.
                            softmax_cross_entropy if output is softmax).
        '''
        
        #Initial parameter values
        self.W_in  = W_in
        self.W_rec = W_rec
        self.W_out = W_out
        self.b_rec = b_rec
        self.b_out = b_out
        
        #Network dimensions
        self.n_in     = W_in.shape[1]
        self.n_hidden = W_in.shape[0]
        self.n_out    = W_out.shape[0]
        
        #Check dimension consistency
        assert self.n_hidden==W_rec.shape[0]
        assert self.n_hidden==W_rec.shape[1]
        assert self.n_hidden==W_in.shape[0]
        assert self.n_hidden==W_out.shape[1]
        assert self.n_hidden==b_rec.shape[0]
        assert self.n_out==b_out.shape[0]
        
        #Define shapes and params lists for convenience later
        self.shapes = [w.shape for w in [W_rec, W_in, b_rec, W_out, b_out]]
        self.params = [self.W_rec, self.W_in, self.b_rec, self.W_out, self.b_out]
        
        #Activation and loss functions
        self.alpha      = alpha
        self.activation = activation
        self.output     = output
        self.loss       = loss
        
        #Number of parameters
        self.n_hidden_params = self.W_rec.size +\
                               self.W_in.size  +\
                               self.b_rec.size
        self.n_params        = self.n_hidden_params +\
                               self.W_out.size +\
                               self.b_out.size
        
        #Params for L2 regularization
        self.L2_indices = [0, 1, 3]
        
        #Initial state values
        self.reset_network()
        
    def reset_network(self, **kwargs):
        
        if 'h' in kwargs.keys():
            self.h = kwargs['h']
        else:
            self.h = np.random.normal(0, 1/np.sqrt(self.n_hidden), self.n_hidden)
            
        self.a = self.activation.f(self.h)
        
        if 'a' in kwargs.keys():
            self.a = kwargs['a']
        
        self.z = self.W_out.dot(self.a) + self.b_out
        
    def next_state(self, x, a=None, update=True, sigma=0):
        '''
        Accepts as argument the current time step's input x and updates
        the state of the RNN, while storing the previous state h
        and activatation a.
        '''
           
        if update:
            if type(x) is np.ndarray:
                self.x = x
            else:
                self.x = np.array([x])
            
            self.h_prev = np.copy(self.h)
            self.a_prev = np.copy(self.a)
            
            self.noise = np.random.normal(0, sigma, self.n_hidden)
            self.h = self.W_rec.dot(self.a) + self.W_in.dot(self.x) + self.b_rec + self.noise
            self.a = (1 - self.alpha)*self.a + self.alpha*self.activation.f(self.h)
        else:
            noise = np.random.normal(0, sigma, self.n_hidden)
            h = self.W_rec.dot(a) + self.W_in.dot(x) + self.b_rec + noise
            return (1 - self.alpha)*a + self.alpha*self.activation.f(h)

    def z_out(self):
        '''
        Updates the output of the RNN using the current activations
        '''
        
        self.z_prev = np.copy(self.z)
        self.z = self.W_out.dot(self.a) + self.b_out
            
    def get_a_jacobian(self, update=True, **kwargs):
        '''
        By default, it updates the Jacobian of the network,
        self.a_J, to the value based on the current parameter
        values and pre-activations. If update=False, then
        it does *not* update self.a_J, but rather returns
        the Jacobian calculated from current pre-activation
        values. If a keyword argument for 'h' or 'W_rec' is
        provided, these arguments are used instead of the
        network's current values.
        '''
        
        #Use kwargs instead of defaults if provided
        try:
            h = kwargs['h']
        except KeyError:
            h = np.copy(self.h)
        try:
            W_rec = kwargs['W_rec']
        except KeyError:
            W_rec = np.copy(self.W_rec)
        
        #Element-wise nonlinearity derivative
        D = self.activation.f_prime(h)
        a_J = self.alpha*np.diag(D).dot(W_rec) + (1 - self.alpha)*np.eye(self.n_hidden)
        
        if update:
            self.a_J = np.copy(a_J)
        else:
            return a_J
        
        
        
        
    
