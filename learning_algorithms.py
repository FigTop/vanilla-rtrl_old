#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 15:03:11 2018

@author: omarschall
"""

import numpy as np
from pdb import set_trace

class BPTT:
    
    def __init__(self, net, t1, t2):
        
        #RNN instance that BPTT is being applied to
        self.net = net
        
        #The two integer parameters
        self.t1  = t1    #Number of steps to average loss over
        self.t2  = t2    #Number of steps to propagate backwards
        self.T   = t1+t2 #Total amount of net hist needed to memorize
        
        #Lists storing relevant net history
        self.h_hist = [np.zeros(self.net.n_hidden) for _ in range(self.T)]
        self.a_hist = [np.zeros(self.net.n_hidden) for _ in range(self.T)]
        self.e_hist = [np.zeros(self.net.n_out) for _ in range(t1)]
        self.x_hist = [np.zeros(self.net.n_in) for _ in range(self.T)]
        
    def update_learning_vars(self):
        '''
        Must be run every time step to update storage of relevant history
        '''
        
        for attr in ['h', 'a', 'e', 'x']:
            self.__dict__[attr+'_hist'].append(getattr(self.net, attr))
            del(self.__dict__[attr+'_hist'][0])
    
    def __call__(self):
        '''
        Run only when grads are needed for the optimizer
        '''
        
        W_out_grad = [np.multiply.outer(self.e_hist[-i],self.a_hist[-i]) for i in range(1, self.t1+1)]
        W_out_grad = sum(W_out_grad)/self.t1
        
        b_out_grad = [self.e_hist[-i] for i in range(1, self.t1+1)]
        b_out_grad = sum(b_out_grad)/self.t1
        
        outer_grads = [W_out_grad, b_out_grad]
        
        get_a_J = self.net.get_a_jacobian
        a_Js = [get_a_J(update=False, h=self.h_hist[-i], h_prev=self.h_hist[-(i+1)]) for i in range(1, self.T)]
        
        n_hidden, n_in = self.net.n_hidden, self.net.n_in
        rec_grad = np.zeros((n_hidden, n_hidden+n_in+1))
        
        for i in range(1, self.t1+1):
            
            q = self.e_hist[-i].dot(self.net.W_out)
            
            for j in range(self.t2):
                
                J = np.eye(n_hidden)
                
                for k in range(j):
                    
                    J = J.dot(a_Js[-(i+k)])
                
                J = J.dot(np.diag(self.net.activation.f_prime(self.h_hist[-(i+j)])))
                
                pre_activity = np.concatenate([self.h_hist[-(i+j+1)], self.x_hist[-(i+j+1)], np.array([1])])
                rec_grad += np.multiply.outer(q.dot(J), pre_activity)
        
        grads = [rec_grad[:,:n_hidden], rec_grad[:,n_hidden:-1], rec_grad[:,-1]] + outer_grads
        
        return grads
        
class RTRL:

    def __init__(self, net):
        
        #Network we're training
        self.net   = net
        
        #Total post- and pre-synaptic units in hidden layer, counting inputs and bias
        self.I     = self.net.n_hidden
        self.J     = self.net.n_hidden + self.net.n_in + 1
        
        #Initialize influence matrix
        self.dadw  = np.zeros((self.net.n_hidden, self.net.n_hidden_params))
        
    def update_learning_vars(self):
        
        self.a_hat = np.concatenate([self.net.a_prev, self.net.x, np.array([1])])
        self.partial_a_partial_w = np.kron(self.a_hat, np.diag(self.net.activation.f_prime(self.net.h)))
        
        self.net.get_a_jacobian()
        self.dadw = self.net.a_J.dot(self.dadw) + self.partial_a_partial_w
        
    def __call__(self):
        
        outer_grads = [np.multiply.outer(self.net.e, self.net.a), self.net.e]
        
        q = self.net.e.dot(self.net.W_out)
        rec_grad = q.dot(self.dadw).reshape((self.I, self.J), order='C')
        
        grads = [rec_grad[:,:self.net.n_hidden], rec_grad[:,self.net.n_hidden:-1], rec_grad[:,-1]] + outer_grads
        
        return grads
    
class UORO:
    
    def __init__(self, net):
        
        #Network we're training
        self.net = net
        
        #Total post- and pre-synaptic units in hidden layer, counting inputs and bias
        self.I     = self.net.n_hidden
        self.J     = self.net.n_hidden + self.net.n_in + 1
        
        #Initialize a_tilde and theta_tilde vectors
        self.theta_tilde = np.random.normal(0, 1, net.n_hidden_params)
        self.a_tilde     = np.random.normal(0, 1, net.n_hidden)
        
    def update_learning_vars(self):
        
        self.a_hat = np.concatenate([self.net.a_prev, self.net.x, np.array([1])])
        self.partial_a_partial_w = np.kron(self.a_hat, np.diag(self.net.activation.f_prime(self.net.h)))
        
        self.net.get_a_jacobian()
        
        nu = np.random.uniform(-1, 1, self.net.n_hidden)
        
        p1 = np.sqrt(np.sqrt(np.sum(self.theta_tilde**2)/np.sum((self.net.a_J.dot(self.a_tilde))**2)))
        p2 = np.sqrt(np.sqrt(np.sum((nu.dot(self.partial_a_partial_w))**2)/np.sum((nu)**2)))
        
        self.a_tilde = p1*self.net.a_J.dot(self.a_tilde) + p2*nu
        self.theta_tilde = (1/p1)*self.theta_tilde + (1/p2)*nu.dot(self.partial_a_partial_w)
    
    def __call__(self):
        
        outer_grads = [np.multiply.outer(self.net.e, self.net.a), self.net.e]
        
        q = self.net.e.dot(self.net.W_out)
        rec_grad = (q.dot(self.a_tilde)*self.theta_tilde).reshape((self.I,self.J), order='F')
        
        grads = [rec_grad[:,:self.net.n_hidden], rec_grad[:,self.net.n_hidden:-1], rec_grad[:,-1]] + outer_grads
        
        return grads
    
class KF_RTRL:
    
    def __init__(self, net):
        
        #Network we're training
        self.net = net
        
        #Total post- and pre-synaptic units in hidden layer, counting inputs and bias
        self.I     = self.net.n_hidden
        self.J     = self.net.n_hidden + self.net.n_in + 1
        
        #Initialize A and u matrices
        self.A = np.random.normal(0, 1/np.sqrt(self.I), (self.I, self.I))
        self.u = np.random.normal(0, 1, self.J)
        
    def update_learning_vars(self):
        
        #Get updated jacobian
        self.net.get_a_jacobian()
        
        #Define necessary components
        self.a_hat   = np.concatenate([self.net.a_prev, self.net.x, np.array([1])])
        self.D       = np.diag(self.net.activation.f_prime(self.net.h))
        self.H_prime = self.net.a_J.dot(self.A)
        
        self.c1, self.c2 = np.random.uniform(-1, 1, 2)
        self.p1          = np.sqrt(np.sqrt(np.sum(self.H_prime**2)/np.sum(self.u**2)))
        self.p2          = np.sqrt(np.sqrt(np.sum(self.D**2)/np.sum(self.a_hat**2)))
        
        self.u = self.c1*self.p1*self.u + self.c2*self.p2*self.a_hat
        self.A = self.c1*(1/self.p1)*self.H_prime + self.c2*(1/self.p2)*self.D
        
    def __call__(self):
        
        outer_grads = [np.multiply.outer(self.net.e, self.net.a), self.net.e]
        
        q = self.net.e.dot(self.net.W_out)
        rec_grad = np.kron(self.u, q.dot(self.A)).reshape((self.I,self.J), order='C')
        
        grads = [rec_grad[:,:self.net.n_hidden], rec_grad[:,self.net.n_hidden:-1], rec_grad[:,-1]] + outer_grads
        
        return grads
    
class DNI:
    
    def __init__(self, net, optimizer):
        
        self.net = net
        self.optimizer = optimizer
        
        self.A = np.random.normal(0, 1/np.sqrt(self.n_hidden + self.n_hidden), (self.n_hidden, self.n_hidden))
        self.B = np.random.normal(0, 1/np.sqrt(self.n_hidden + self.n_out), (self.n_hidden, self.n_out))
        self.C = np.zeros(self.n_hidden)
        
    def update_learning_vars(self):
        
        self.sg_1 = self.synthetic_grad(self.a_prev, self.y_prev)
        self.sg_2 = (1 - self.alpha_SG_target)*self.sg_2 + self.synthetic_grad(self.a, self.y).dot(self.a_J)
        self.e_sg = self.sg_1 - self.sg_2
        
        self.SG_grads = [np.multiply.outer(self.e_sg, self.a_prev) + self.l2_SG*self.A,
                     np.multiply.outer(self.e_sg, self.y_prev) + self.l2_SG*self.B,
                     self.e_sg]
        self.SG_params = self.SG_optimizer.get_update(self.SG_params, self.SG_grads)
        
        self.A, self.B, self.C = self.SG_params
        
        self.sg_1 = self.synthetic_grad(self.a_prev, self.y_prev)
        self.e_sg = self.sg_1 - self.sg_2
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    