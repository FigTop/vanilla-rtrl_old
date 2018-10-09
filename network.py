#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  7 17:20:39 2018

@author: omarschall
"""

import numpy as np
from utils import *
from optimizers import *

class RNN:
    
    def __init__(self, W_in, W_rec, W_out, b_rec, b_out, activation, alpha, output, loss, **kwargs):
        '''
        Initializes a vanilla RNN object that follows the forward equation
        
        h_t = (1 - alpha)*h_{t-1} + W_rec * phi(h_{t-1}) + W_in * x_t + b_rec
        z_t = W_out * a_t + b_out
        
        with initial parameter values given by W_in, W_rec, W_out, b_rec, b_in
        and specified activation and loss functions, which must be function
        objects--see utils.py.
        '''
        
        allowed_kwargs = {'dadw', 'theta_tilde', 'a_tilde', 'A', 'B', 'C', 'u'}
        for k in kwargs:
            if k not in allowed_kwargs:
                raise TypeError('Unexpected keyword argument '
                                'passed to RNN.__init__: ' + str(k))
        
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
        
        #Initial state values
        self.reset_network()
        self.learning_vars_init()
        self.__dict__.update(kwargs)
        
    def learning_vars_init(self):
        
        #Standard RTRL
        self.dadw  = np.random.normal(0, 1, (self.n_hidden, self.n_hidden_params))     
        #UORO
        self.theta_tilde = np.random.normal(0, 1, self.n_hidden_params)
        self.a_tilde     = np.random.normal(0, 1, self.n_hidden)
        #KF
        self.A = np.random.normal(0, 1/np.sqrt(self.n_hidden + self.n_hidden), (self.n_hidden, self.n_hidden))
        self.u = np.random.normal(0, 1, (self.n_hidden + self.n_in + 1))
        #Synthetic Grads
        self.B = np.random.normal(0, 1/np.sqrt(self.n_hidden + self.n_out), (self.n_hidden, self.n_out))
        self.C = np.zeros(self.n_hidden)
        
    def reset_network(self, **kwargs):
        
        if 'h' in kwargs.keys():
            self.h = kwargs['h']
        else:
            self.h = np.random.normal(0, 1/np.sqrt(self.n_hidden), self.n_hidden)
            
        self.a = self.activation.f(self.h)
        self.z = self.W_out.dot(self.a) + self.b_out
        
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
        
        self.h = (1 - self.alpha)*self.h + self.W_rec.dot(self.a) + self.W_in.dot(self.x) + self.b_rec
        self.a = self.activation.f(self.h)
        
    def z_out(self):
        
        self.z_prev = np.copy(self.z)
        self.z = self.W_out.dot(self.a) + self.b_out
            
    def get_a_jacobian(self, update=True, **kwargs):
        
        try:
            h = kwargs['h']
        except KeyError:
            h = np.copy(self.h)
        
        try:
            h_prev = kwargs['h_prev']
        except KeyError:
            h_prev = np.copy(self.h_prev)
        
        q1 = self.activation.f_prime(h)
        
        if self.alpha!=1:
            q2 = self.activation.f_prime(h_prev)
            a_J = np.diag(q1).dot(self.W_rec + np.diag((1-self.alpha)/(q2)))
        else:
            a_J = np.diag(q1).dot(self.W_rec)
        
        if update:
            self.a_J = np.copy(a_J)
        else:
            return a_J
    
    def get_partial_a_partial_w(self):

        self.a_hat = np.concatenate([self.a_prev, self.x, np.array([1])])
        self.partial_a_partial_w = np.kron(self.a_hat, np.diag(self.activation.f_prime(self.h)))
    
    def update_learning_vars(self, method='rtrl'):
        
        assert method in ['rtrl', 'uoro', 'kf', 'dni']
        
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
            self.a_hat   = np.concatenate([self.a_prev, self.x, np.array([1])])
            self.D       = np.diag(self.activation.f_prime(self.h))
            self.H_prime = self.a_J.dot(self.A)
            
            self.c1, self.c2 = np.random.uniform(-1, 1, 2)
            self.p1          = np.sqrt(np.sqrt(np.sum(self.H_prime**2)/np.sum(self.u**2)))
            self.p2          = np.sqrt(np.sqrt(np.sum(self.D**2)/np.sum(self.a_hat**2)))
            
            self.u = self.c1*self.p1*self.u + self.c2*self.p2*self.a_hat
            self.A = self.c1*(1/self.p1)*self.H_prime + self.c2*(1/self.p2)*self.D
            
        if method=='dni':
            
            self.sg_1 = self.synthetic_grad(self.a_prev, self.y_prev)
            self.sg_2 = (1 - self.alpha_SG_target)*self.sg_2 + self.synthetic_grad(self.a, self.y).dot(self.a_J)
            self.e_sg = self.sg_1 - self.sg_2
            
            for _ in range(self.n_SG):
                self.SG_grads = [np.multiply.outer(self.e_sg, self.a_prev) + self.l2_SG*self.A,
                                 np.multiply.outer(self.e_sg, self.y_prev) + self.l2_SG*self.B,
                                 self.e_sg]
                self.SG_params = self.SG_optimizer.get_update(self.SG_params, self.SG_grads)
                
                self.A, self.B, self.C = self.SG_params
                
                self.sg_1 = self.synthetic_grad(self.a_prev, self.y_prev)
                self.e_sg = self.sg_1 - self.sg_2
            #self.A = self.A - self.SG_learning_rate*(np.multiply.outer(self.e_sg, self.a) + self.l2_SG*self.A)
            #self.B = self.B - self.SG_learning_rate*(np.multiply.outer(self.e_sg, self.y) + self.l2_SG*self.B)
            #self.C = self.C - self.SG_learning_rate*self.e_sg
    
    def synthetic_grad(self, a, y):
        
        return self.A.dot(a) + self.B.dot(y) + self.C
            
    def update_params(self, y, optimizer, method='rtrl'):
        
        assert method in ['rtrl', 'uoro', 'kf', 'dni']
        
        #Compute error term via loss derivative for label y
        e = self.loss.f_prime(self.z, y)
        self.e = e
        
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
            
        if method=='dni':
            sg = self.synthetic_grad(self.a, self.y)*self.activation.f_prime(self.h)
            a_ext = np.concatenate([self.a, self.x, np.array([1])])
            gradient = np.concatenate([np.kron(a_ext, sg)]+outer_grads)

        #Reshape gradient into correct sizes
        grads = [gradient[self.flat_idx[i]:self.flat_idx[i+1]].reshape(s, order='F') for i, s in enumerate(self.shapes)]
        
        if self.l2_reg>0:
            for i in [0, 1, 3]:
                grads[i] += self.l2_reg*self.params[i]
        
        self.grads = grads
        
        #Use optimizer object to update parameters
        self.params = optimizer.get_update(self.params, grads)
        self.W_rec, self.W_in, self.b_rec, self.W_out, self.b_out = self.params
        
    def run(self, x_inputs, y_labels, learning_algorithm, optimizer, method='rtrl', **kwargs):
        
        allowed_kwargs = {'l2_reg', 'l2_SG', 't_stop_learning', 'monitors', 'SG_optimizer',
                          'alpha_SG_target', 'n_SG'}
        for k in kwargs:
            if k not in allowed_kwargs:
                raise TypeError('Unexpected keyword argument '
                                'passed to RNN.run: ' + str(k))
        
        #Default run parameters
        self.t_stop_learning  = len(x_inputs)
        self.l2_reg           = 0
        self.SG_l2_reg        = 0
        self.SG_learning_rate = 0.001
        self.SG_clipnorm      = 1.0
        self.SG_optimizer     = SGD(lr=self.SG_learning_rate, clipnorm=self.SG_clipnorm)
        self.monitors         = []
        self.t_update_ratio   = 1
    
        self.__dict__.update(kwargs)
        self.reset_network()
        
        if method=='dni':
            self.SG_params = [self.A, self.B, self.C]
            self.sg_1 = np.zeros(self.n_hidden)
            self.sg_2 = np.zeros(self.n_hidden)
            self.SG_learning_rate = SG_learning_rate
            if not hasattr(self, 'alpha_SG_target'):
                self.alpha_SG_target = 0.1
            if not hasattr(self, 'n_SG'):
                self.n_SG = 1
        
        #Initialize monitors
        self.mons = {}
        for mon in self.monitors:
            self.mons[mon] = []
        
        self.y_prev = y_labels[0]
        
        for i_t in range(len(x_inputs)):
            
            self.x = x_inputs[i_t]
            self.y = y_labels[i_t]
            
            self.next_state(self.x)
            self.z_out()
            
            self.y_hat  = self.output.f(self.z)
            self.loss_  = self.loss.f(self.z, self.y)
            
            if i_t < t_stop_learning:
                self.get_a_jacobian()
                self.update_learning_vars(method=method)
                self.update_params(self.y, optimizer=optimizer, method=method)
                
            for key in self.mons.keys():
                self.mons[key].append(getattr(self, key))
                
            self.y_prev = np.copy(self.y)
                
            
        #return losses, y_hats
            
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    