#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  7 17:20:39 2018

@author: omarschall
"""

import numpy as np
from utils import *

class RNN():
    
    def __init__(self, W_in, W_rec, W_out, b_rec, b_out, activation, loss):
        
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
        
        #Activation function
        self.activation = activation
        
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
        self.dhdw  = np.zeros((self.n_hidden, self.n_hidden_params))     
        #UORO
        self.theta_tilde = np.zeros(self.n_hidden_params)
        self.h_tilde     = np.zeros(self.n_hidden)
        #KF
        self.A = np.random.normal(0, 1, (self.n_hidden, self.n_hidden))
        self.u = np.random.normal(0, 1, (self.n_hidden + self.n_in + 1))
        
        #Loss function
        self.loss = loss
        
    def next_state(self, x):
        
        if type(x) is np.ndarray:
            self.x = x
        else:
            self.x = np.array([x])
        
        self.h_prev = np.copy(self.h)
        self.a_prev = self.activation.f(self.h_prev)
        
        self.h = self.W_rec.dot(self.a) + self.W_in.dot(self.x) + self.b_rec
        self.a = self.activation.f(self.h)
        
    def z_out(self):
        
        self.z = self.W_out.dot(self.h) + self.b_out
        
    def reset_network(self, h=None):
        
        if h is not None:
            self.h = h
        else:
            self.h = np.random.normal(0, 1/np.sqrt(self.n_hidden), self.n_hidden)
            
        self.a = self.activation.f(self.h)
            
    def get_h_jacobian(self):
        
        self.h_J = self.W_rec.dot(np.diag(self.activation.f_prime(self.h_prev)))
    
    def get_partial_h_partial_w(self):
        
        dhdw_rec = np.kron(self.a, np.eye(self.n_hidden))
        dhdw_in  = np.kron(self.x, np.eye(self.n_hidden))
        dhdb_rec = np.eye(self.n_hidden)
        
        self.partial_h_partial_w = np.concatenate([dhdw_rec, dhdw_in, dhdb_rec], axis=1)
    
    def update_dhdw(self, method='rtrl'):
        
        assert method in ['rtrl', 'uoro', 'kf']
        
        if method=='rtrl':
            
            self.dhdw = self.h_J.dot(self.dhdw) + self.partial_h_partial_w
        
        if method=='uoro':
            
            pass
            #self.theta_tilde
            #self.h_tilde
        
        if method=='kf':
            
            #Define necessary components
            h_hat   = np.concatenate([self.h_prev, self.x, np.array([1])])
            D       = np.diag(self.activation.f_prime(self.h_prev))
            H_prime = self.h_J.dot(self.A)
            
            c1, c2 = np.random.uniform(-1, 1, 2)
            p1 = np.sqrt(np.sqrt(np.sum(H_prime**2)/np.sum(self.u**2)))
            p2 = np.sqrt(np.sqrt(np.sum(D**2)/np.sum(h_hat**2)))
            
            self.u = c1*p1*self.u + c2*p2*h_hat
            self.A = c1*(1/p1)*H_prime + c2*(1/p2)*D
            
            
    def update_params(self, y, learning_rate=0.001, method='rtrl'):
        
        assert method in ['rtrl', 'uoro', 'kf']
        
        #Compute error term via loss derivative for label y
        e = self.loss.f_prime(self.z, y)
        
        #Compute the dependence of the error on the hidden state
        try:
            q = e.dot(self.W_out)
        except AttributeError: #In case e is a scalar
            q = np.array([e]).dot(self.W_out)
        
        
        #Calculate the gradient using preferred method
        if method=='rtrl':
            gradient = q.dot(self.dhdw)
            
        if method=='uoro':
            gradient = q.dot(self.h_tilde)*self.theta_tilde
            
        if method=='kf':
            gradient = np.kron(self.u, q.dot(self.A))
            
        n = np.cumsum([w.size for w in [self.W_rec,
                                        self.W_in,
                                        self.b_rec]])
        
        
        #Update "hidden" parameters
        self.W_rec -= learning_rate*gradient[0:n[0]].reshape((self.n_hidden, self.n_hidden))
        self.W_in  -= learning_rate*gradient[n[0]:n[1]].reshape((self.n_hidden, self.n_in))
        self.b_rec -= learning_rate*gradient[n[1]:n[2]]

        #Update "outer" parameters
        self.W_out -= learning_rate*np.multiply.outer(e, self.h)
        self.b_out -= learning_rate*e
        
    def run(self, x_inputs, y_labels, learning_rate=0.001, method='rtrl'):
        
        self.reset_network()
        
        losses = []
        y_hats = []
        
        for i_t in range(len(x_inputs)):
            
            x = x_inputs[i_t]
            y = y_labels[i_t]
            
            self.next_state(x)
            self.z_out()
            
            #TODO make this more general
            y_hat = sigmoid.f(self.z)
            losses.append(self.loss.f(self.z, y))
            y_hats.append(y_hat)
            
            self.get_h_jacobian()
            self.get_partial_h_partial_w()
            self.update_dhdw(method=method)
            self.update_params(y, learning_rate=learning_rate, method=method)
            
        return losses, y_hats
            
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    