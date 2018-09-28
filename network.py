#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  7 17:20:39 2018

@author: omarschall
"""

import numpy as np
from utils import *

class RNN:
    
    def __init__(self, W_in, W_rec, W_out, b_rec, b_out, activation, output, loss):
        '''
        Initializes a vanilla RNN object that follows the forward equation
        
        h_t = W_rec * phi(h_{t-1}) + W_in * x_t + b_rec
        z_t = W_out * h_t + b_out
        
        with initial parameter values given by W_in, W_rec, W_out, b_rec, b_in
        and specified activation and loss functions, which must be function
        objects--see utils.py.
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
        self.flat_idx = np.cumsum([0]+[w.size for w in self.params])
        
        #Activation and loss functions
        self.activation = activation
        self.output     = output
        self.loss       = loss
        
        #Number of parameters
        self.n_params = self.W_rec.size +\
                        self.W_in.size  +\
                        self.b_rec.size +\
                        self.W_out.size +\
                        self.b_out.size
                        
        #Number of parameters for hidden layer
        self.n_hidden_params = self.W_rec.size +\
                               self.W_in.size  +\
                               self.b_rec.size
        
        #Initial state values
        self.h = np.random.normal(0, 1/np.sqrt(self.n_hidden), self.n_hidden)
        self.a = self.activation.f(self.h)

        #Standard RTRL
        self.dadw  = np.random.normal(0, 1, (self.n_hidden, self.n_hidden_params))     
        #UORO
        self.theta_tilde = np.random.normal(0, 1, self.n_hidden_params)
        self.a_tilde     = np.random.normal(0, 1, self.n_hidden)
        #KF
        self.A = np.random.normal(0, 1, (self.n_hidden, self.n_hidden))
        self.u = np.random.normal(0, 1, (self.n_hidden + self.n_in + 1))
        
    def next_state(self, x):
        '''
        Accepts as argument the current time step's input x and updates
        the state of the RNN, while storing the previous state h
        and activatation a.
        '''
        
        
        if type(x) is np.ndarray:
            self.x = x
        else:
            self.x = np.array([x])
        
        self.h_prev = np.copy(self.h)
        self.a_prev = np.copy(self.a)
        
        self.h = self.W_rec.dot(self.a) + self.W_in.dot(self.x) + self.b_rec
        self.a = self.activation.f(self.h)
        
    def z_out(self):
        
        self.z = self.W_out.dot(self.a) + self.b_out
        
    def reset_network(self, h=None):
        
        if h is not None:
            self.h = h
        else:
            self.h = np.random.normal(0, 1/np.sqrt(self.n_hidden), self.n_hidden)
            
        self.a = self.activation.f(self.h)
            
    def get_a_jacobian(self):
        
        self.a_J = np.diag(self.activation.f_prime(self.h)).dot(self.W_rec)
    
    def get_partial_a_partial_w(self):

        a_hat = np.concatenate([self.a_prev, self.x, np.array([1])])
        self.partial_a_partial_w = np.kron(a_hat, np.diag(self.activation.f_prime(self.h)))
    
    def update_dadw(self, method='rtrl'):
        
        assert method in ['rtrl', 'uoro', 'kf']
        
        if method=='rtrl':
            
            self.get_partial_a_partial_w()
            self.dadw = self.a_J.dot(self.dadw) + self.partial_a_partial_w
        
        if method=='uoro':
            
            self.get_partial_a_partial_w()
            
            nu = np.random.uniform(-1, 1, self.n_hidden)
            
            p1 = np.sqrt(np.sqrt(np.sum(self.theta_tilde**2)/np.sum((self.a_J.dot(self.a_tilde))**2)))
            p2 = np.sqrt(np.sqrt(np.sum((nu.dot(self.partial_a_partial_w))**2)/np.sum((nu)**2)))
            
            self.a_tilde = p1*self.a_J.dot(self.a_tilde) + p2*nu
            self.theta_tilde = (1/p1)*self.theta_tilde + (1/p2)*nu.dot(self.partial_a_partial_w)
        
        if method=='kf':
            
            #Define necessary components
            a_hat   = np.concatenate([self.a_prev, self.x, np.array([1])])
            D       = np.diag(self.activation.f_prime(self.h))
            H_prime = self.a_J.dot(self.A)
            
            c1, c2 = np.random.uniform(-1, 1, 2)
            p1 = np.sqrt(np.sqrt(np.sum(H_prime**2)/np.sum(self.u**2)))
            p2 = np.sqrt(np.sqrt(np.sum(D**2)/np.sum(a_hat**2)))
            
            self.u = c1*p1*self.u + c2*p2*a_hat
            self.A = c1*(1/p1)*H_prime + c2*(1/p2)*D
            
            
    def update_params(self, y, optimizer, method='rtrl'):
        
        assert method in ['rtrl', 'uoro', 'kf']
        
        #Compute error term via loss derivative for label y
        e = self.loss.f_prime(self.z, y)
        
        #Compute the dependence of the error on the previous activation
        try:
            q = (e.dot(self.W_out))#.dot(self.W_rec)#.dot(np.diag(self.activation.f_prime(self.h)))
        except AttributeError: #In case e is a scalar
            q = (np.array([e]).dot(self.W_out))#s.dot(self.W_rec)#.dot(np.diag(self.activation.f_prime(self.h)))
        
        
        outer_grads = [np.multiply.outer(e, self.a).flatten(), e]
        
        #Calculate the gradient using preferred method
        if method=='rtrl':
            gradient = np.concatenate([q.dot(self.dadw)]+outer_grads)
            
        if method=='uoro':
            gradient = np.concatenate([q.dot(self.a_tilde)*self.theta_tilde]+outer_grads)
            
        if method=='kf':
            gradient = np.concatenate([np.kron(self.u, q.dot(self.A))]+outer_grads)

        #Reshape gradient into correct sizes
        grads = [gradient[self.flat_idx[i]:self.flat_idx[i+1]].reshape(s, order='F') for i, s in enumerate(self.shapes)]
        
        if self.l2_reg>0:
            for i in [0, 1, 3]:
                grads[i] += self.l2_reg*self.params[i]
        
        self.grads = grads
        
        #Use optimizer object to update parameters
        self.params = optimizer.get_update(self.params, grads)
        self.W_rec, self.W_in, self.b_rec, self.W_out, self.b_out = self.params
        
    def run(self, x_inputs, y_labels, optimizer, method='rtrl', **kwargs):
        
        allowed_kwargs = {'l2_reg', 't_stop_learning'}
        for k in kwargs:
            if k not in allowed_kwargs:
                raise TypeError('Unexpected keyword argument '
                                'passed to self.run: ' + str(k))
        
        self.__dict__.update(kwargs)
        self.reset_network()
        
        if hasattr(self, 't_stop_learning'):
            t_stop_learning = self.t_stop_learning
        else:
            t_stop_learning = len(x_inputs)
            
        if not hasattr(self, 'l2_reg'):
            self.l2_reg = 0
        
        losses = []
        y_hats = []
        
        for i_t in range(len(x_inputs)):
            
            x = x_inputs[i_t]
            y = y_labels[i_t]
            
            self.next_state(x)
            self.z_out()
            
            y_hat = self.output.f(self.z)
            losses.append(self.loss.f(self.z, y))
            y_hats.append(y_hat)
            
            if i_t < t_stop_learning:
                self.get_a_jacobian()
                self.update_dadw(method=method)
                self.update_params(y, optimizer=optimizer, method=method)
            
        return losses, y_hats
            
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    