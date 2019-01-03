#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 15:03:11 2018

@author: omarschall
"""

import numpy as np
from pdb import set_trace
from utils import *

class Learning_Algorithm:
    
    def __init__(self, net, allowed_kwargs_, **kwargs):
        
        allowed_kwargs = {'W_FB'}.union(allowed_kwargs_)
                
        for k in kwargs:
            if k not in allowed_kwargs:
                raise TypeError('Unexpected keyword argument '
                                'passed to Learning_Algorithm.__init__: ' + str(k))
        
        self.__dict__.update(kwargs)
        
        #RNN instance the algorithm is being applied to
        self.net = net
        self.n_in  = self.net.n_in
        self.n_h   = self.net.n_hidden
        self.n_out = self.net.n_out
      
        
class Real_Time_Learning_Algorithm(Learning_Algorithm):
    
    def get_outer_grads(self):
        
        return [np.multiply.outer(self.net.e, self.net.a), self.net.e]
        
    def propagate_feedback_to_hidden(self):
        
        if not hasattr(self, 'W_FB'):
            self.q = self.net.e.dot(self.net.W_out)
        else:
            self.q = self.net.e.dot(self.W_FB)
    
    def __call__(self):
        
        outer_grads = self.get_outer_grads()
        self.propagate_feedback_to_hidden()
        rec_grads = self.get_rec_grads()
        grads = [rec_grads[:,:self.n_h], rec_grads[:,self.n_h:-1], rec_grads[:,-1]] + outer_grads
        
        return grads
        
class RTRL(Real_Time_Learning_Algorithm):

    def __init__(self, net, **kwargs):
        
        allowed_kwargs_ = set()
        super().__init__(net, allowed_kwargs_, **kwargs)
        
        #Initialize influence matrix
        self.dadw  = np.zeros((self.n_h, self.net.n_hidden_params))
        
    def update_learning_vars(self):
        
        self.a_hat = np.concatenate([self.net.a_prev, self.net.x, np.array([1])])
        self.papw = np.kron(self.a_hat, np.diag(self.net.activation.f_prime(self.net.h)))
        self.net.get_a_jacobian()
        self.dadw = self.net.a_J.dot(self.dadw) + self.papw
        
    def get_rec_grads(self):

        return self.q.dot(self.dadw).reshape((self.n_h, self.n_h + self.n_in + 1), order='F')
    
class UORO(Real_Time_Learning_Algorithm):
    
    def __init__(self, net, **kwargs):
        
        allowed_kwargs_ = {'epsilon', 'P1', 'P2'}
        super().__init__(net, allowed_kwargs_, **kwargs)
        
        #Initialize a_tilde and theta_tilde vectors
        self.theta_tilde = np.random.normal(0, 1, net.n_hidden_params)
        self.a_tilde     = np.random.normal(0, 1, net.n_hidden)
        
    def update_learning_vars(self):
        
        self.a_hat = np.concatenate([self.net.a_prev, self.net.x, np.array([1])])
        self.papw = np.kron(self.a_hat, np.diag(self.net.activation.f_prime(self.net.h)))
        
        self.net.get_a_jacobian()
        
        self.nu = np.random.uniform(-1, 1, self.net.n_hidden)
        
        #Forward differentiation method
        if hasattr(self, 'epsilon'):
            self.a_eps = self.net.a + self.epsilon*self.a_tilde
            self.f1 = self.net.next_state(self.net.x, self.a_eps, update=False)  
            self.f2 = self.net.next_state(self.net.x, self.net.a, update=False)
            self.a_tilde_ = (self.f1 - self.f2)/self.epsilon
            self.p1 = np.sqrt(np.sqrt(np.sum(self.theta_tilde**2))/(np.sqrt(np.sum(self.a_tilde_**2)) + self.epsilon)) + self.epsilon
            self.p2 = np.sqrt(np.sqrt(np.sum((nu.dot(self.partial_a_partial_w))**2))/(np.sqrt(np.sum(nu**2)) + self.epsilon)) + self.epsilon

        #Compute normalizers
        self.p1 = np.copy(self.P1)
        self.p2 = np.copy(self.P2)
        
        if self.P1 is None:
            self.p1 = np.sqrt(np.sqrt(np.sum(self.theta_tilde**2))/(np.sqrt(np.sum(self.a_tilde_**2)) + self.epsilon)) + self.epsilon
        if self.P2 is None:
            self.p2 = np.sqrt(np.sqrt(np.sum((nu.dot(self.partial_a_partial_w))**2))/(np.sqrt(np.sum(nu**2)) + self.epsilon)) + self.epsilon
        
        #self.a_tilde = p1*self.net.a_J.dot(self.a_tilde) + p2*nu
        self.a_tilde = self.p1*self.a_tilde_ + self.p2*nu
        self.theta_tilde = (1/self.p1)*self.theta_tilde + (1/self.p2)*nu.dot(self.partial_a_partial_w)
    
    def get_rec_grads(self):
        
        self.Q = self.q.dot(self.a_tilde) #"Global learning signal"
        return (self.Q*self.theta_tilde).reshape((self.n_h, self.n_h + self.n_in + 1), order='F')
    
class KF_RTRL(Real_Time_Learning_Algorithm):
    
    def __init__(self, net, monitors=[]):
        
        super().__init__(net, monitors)
        
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
        #self.p1 = 0.8
        #self.p2 = 1.1
        
        self.u = self.c1*self.p1*self.u + self.c2*self.p2*self.a_hat
        self.A = self.c1*(1/self.p1)*self.H_prime + self.c2*(1/self.p2)*self.D
        
    def __call__(self):
        
        outer_grads = [np.multiply.outer(self.net.e, self.net.a), self.net.e]
        
        q = self.net.e.dot(self.net.W_out)
        rec_grad = np.kron(self.u, q.dot(self.A)).reshape((self.I,self.J), order='F')
        
        grads = [rec_grad[:,:self.net.n_hidden], rec_grad[:,self.net.n_hidden:-1], rec_grad[:,-1]] + outer_grads
        
        self.update_monitors()
        
        return grads
    
class DNI(Real_Time_Learning_Algorithm):
    
    def __init__(self, net, optimizer, monitors=[], activation=identity,
                 lambda_mix=0, l2_reg=0, fix_SG_interval=1, **kwargs):
        
        allowed_kwargs = {'SG_clipnorm', 'SG_target_clipnorm', 'W_a_lr',
                          'SG_label_activation', 'feedback', 'backprop_weights',
                          'sg_loss_thr', 'U_lr'}
        for k in kwargs:
            if k not in allowed_kwargs:
                raise TypeError('Unexpected keyword argument '
                                'passed to RNN.run: ' + str(k))
        
        super().__init__(net, monitors)
        
        self.optimizer = optimizer
        self.activation = activation
        self.lambda_mix = lambda_mix
        self.l2_reg = l2_reg
        self.fix_SG_interval = fix_SG_interval
        self.SG_label_activation = identity
        self.feedback = 'symmetric'
        self.backprop_weights = 'exact'
        self.i_fix = 0
        self.sg_loss_thr = 0.05
        
        n_h = self.net.n_hidden
        n_out = self.net.n_out
        
        self.A = np.random.normal(0, 1/np.sqrt(n_h), (n_h, n_h))
        self.B = np.random.normal(0, 1/np.sqrt(n_out), (n_h, n_out))
        self.C = np.zeros(n_h)
        
        self.W_a = np.copy(self.net.W_rec)
        #self.W_a = self.net.W_rec
        self.W_fb = np.random.normal(0, 1/np.sqrt(n_h), (n_out, n_h))
        self.U = np.copy(self.A)
        
        self.A_, self.B_, self.C_ = np.copy(self.A), np.copy(self.B), np.copy(self.C)
        
        self.SG_params = [self.A, self.B, self.C]
        
        self.__dict__.update(kwargs)
        
    def update_learning_vars(self):
        
        #Get network jacobian
        self.net.get_a_jacobian()
        
        #Computer SG error term
        self.sg = self.synthetic_grad(self.net.a_prev, self.net.y_prev)
        
        if hasattr(self, 'SG_clipnorm'):
            self.sg_norm = np.sqrt((self.sg ** 2).sum())
            if self.sg_norm > self.SG_clipnorm:
                self.sg = self.sg / self.sg_norm
                
        self.sg_target = self.get_sg_target()
        
        if hasattr(self, 'SG_target_clipnorm'):
            self.sg_target_norm = np.sqrt((self.sg_target ** 2).sum())
            if self.sg_target_norm > self.SG_target_clipnorm:
                self.sg_target = self.sg_target / self.sg_target_norm
            
        self.e_sg = self.sg - self.sg_target
        self.sg_loss = np.mean((self.sg - self.sg_target)**2)
        self.scaled_e_sg = self.e_sg*self.activation.f_prime(self.sg_h)
        
        #Get SG grads
        self.SG_grads = [np.multiply.outer(self.scaled_e_sg, self.net.a_prev),
                         np.multiply.outer(self.scaled_e_sg, self.net.y_prev),
                         self.scaled_e_sg]
        
        if self.l2_reg > 0:
            self.SG_grads[0] += self.l2_reg*self.A
            self.SG_grads[1] += self.l2_reg*self.B
            self.SG_grads[2] += self.l2_reg*self.C
        
        #Update SG parameters
        self.SG_params = self.optimizer.get_update(self.SG_params, self.SG_grads)
        self.A, self.B, self.C = self.SG_params
        
        if self.i_fix == self.fix_SG_interval - 1:
            self.i_fix = 0
            self.A_, self.B_, self.C_ = np.copy(self.A), np.copy(self.B), np.copy(self.C)
        else:
            self.i_fix += 1
        
        if hasattr(self, 'W_a_lr'):
            self.update_W_a()
            
        if hasattr(self, 'U_lr'):
            self.update_U()
        
    def get_sg_target(self):
        
        try:
            true_grad = self.net.comp_alg.credit_assignment
        except AttributeError:
            true_grad = 0
        
        if self.feedback=='symmetric':
            self.q = self.net.e.dot(self.net.W_out)
        elif self.feedback=='random':
            self.q = self.net.e.dot(self.W_fb)
        
        if self.backprop_weights=='exact':
            bootstrap = self.q + self.synthetic_grad_(self.net.a, self.net.y).dot(self.net.a_J)
        elif self.backprop_weights=='approximate':
            bootstrap = self.q + self.synthetic_grad_(self.net.a, self.net.y).dot(self.W_a)
        elif self.backprop_weights=='composite':
            bootstrap = self.q + self.U.dot(self.net.a)
            
        #bootstrap = self.q + self.synthetic_grad_(self.net.a, self.net.y).dot(np.eye(self.net.n_hidden))
        
        return self.lambda_mix*true_grad + (1 - self.lambda_mix)*bootstrap
    
    def update_W_a(self):
        
        self.loss_a = np.square(self.W_a.dot(self.net.a_prev) - self.net.a).mean()
        self.e_a = self.W_a.dot(self.net.a_prev) - self.net.a
        
        #self.loss_a = np.square(self.net.a_prev.dot(self.W_a) - self.net.a).mean()
        #self.e_a = self.net.a_prev.dot(self.W_a) - self.net.a

        self.W_a -= self.W_a_lr*np.multiply.outer(self.e_a, self.net.a_prev)
        
    def update_U(self):
        
        self.loss_u = np.square(self.U.dot(self.net.a_prev) - self.sg).mean()
        self.e_u = self.U.dot(self.net.a_prev) - self.sg
        self.U -= self.U_lr*np.multiply.outer(self.e_u, self.net.a_prev)
        
    def synthetic_grad(self, a, y):
        self.sg_h = self.A.dot(a) + self.B.dot(y) + self.C
        return self.activation.f(self.sg_h)
        
    def synthetic_grad_(self, a, y):
        self.sg_h_ = self.A_.dot(a) + self.B_.dot(y) + self.C_
        return self.SG_label_activation.f((self.activation.f(self.sg_h_)))
    
    def __call__(self):
        
        outer_grads = [np.multiply.outer(self.net.e, self.net.a), self.net.e]
    
        self.sg = self.synthetic_grad(self.net.a, self.net.y)
        self.sg_scaled = self.sg*self.net.activation.f_prime(self.net.h)
        
        if hasattr(self, 'SG_clipnorm'):
            norm = np.sqrt((self.sg ** 2).sum())
            if norm > self.SG_clipnorm:
                self.sg = self.sg / norm
                
        self.pre_activity = np.concatenate([self.net.a_prev, self.net.x, np.array([1])])
        rec_grad = np.multiply.outer(self.sg_scaled, self.pre_activity)
        
        grads = [rec_grad[:,:self.net.n_hidden], rec_grad[:,self.net.n_hidden:-1], rec_grad[:,-1]] + outer_grads
        
        self.update_monitors()
        
        return grads
    
class RFLO(Learning_Algorithm):
    
    def __init__(self, net, alpha, monitors=[], **kwargs):
        
        allowed_kwargs = {'W_FB', 'P'}
        for k in kwargs:
            if k not in allowed_kwargs:
                raise TypeError('Unexpected keyword argument '
                                'passed to RFLO.__init__: ' + str(k))
        
        super().__init__(net, monitors)
        
        self.alpha = alpha
        
        n_h = self.net.n_hidden
        n_in = self.net.n_in
        self.P = np.zeros((n_h, n_h + n_in + 1))
        
        self.__dict__.update(kwargs)
        
    def update_learning_vars(self):
        
        self.a_hat   = np.concatenate([self.net.a_prev, self.net.x, np.array([1])])
        self.D = self.net.activation.f_prime(self.net.h)
        self.P = (1 - self.alpha)*self.P + self.alpha*np.multiply.outer(self.D, self.a_hat)
        
    def __call__(self):
    
        outer_grads = [np.multiply.outer(self.net.e, self.net.a), self.net.e] 
        
        n_h = self.net.n_hidden
    
        if hasattr(self, 'W_FB'):
            self.q = self.net.e.dot(self.W_FB)
        else:
            self.q = self.net.e.dot(self.W_out)
        rec_grad = (self.q*self.P.T).T
        
        grads = [rec_grad[:,:n_h], rec_grad[:,n_h:-1], rec_grad[:,-1]] + outer_grads
        self.update_monitors()
        
        return grads

class BPTT(Learning_Algorithm):
    
    def __init__(self, net, t1, t2, monitors=[], use_historical_W=False):
    
        super().__init__(net, monitors)
        
        #The two integer parameters
        self.t1  = t1    #Number of steps to average loss over
        self.t2  = t2    #Number of steps to propagate backwards
        self.T   = t1+t2 #Total amount of net hist needed to memorize
        self.use_historical_W = use_historical_W
        
        #Lists storing relevant net history
        self.h_hist = [np.zeros(self.net.n_hidden) for _ in range(self.T)]
        self.a_hist = [np.zeros(self.net.n_hidden) for _ in range(self.T)]
        self.e_hist = [np.zeros(self.net.n_out) for _ in range(t1)]
        self.x_hist = [np.zeros(self.net.n_in) for _ in range(self.T)]
        self.W_rec_hist = [self.net.W_rec for _ in range(self.T)]
        
    def update_learning_vars(self):
        '''
        Must be run every time step to update storage of relevant history
        '''
        
        for attr in ['h', 'a', 'e', 'x', 'W_rec']:
            self.__dict__[attr+'_hist'].append(getattr(self.net, attr))
            del(self.__dict__[attr+'_hist'][0])
    
    def __call__(self):
        '''
        Run only when grads are needed for the optimizer
        '''
        
        W_out_grad = [np.multiply.outer(self.e_hist[-i],self.a_hist[-i]) for i in range(1, self.t1+1)]
        self.W_out_grad = sum(W_out_grad)/self.t1
        
        b_out_grad = [self.e_hist[-i] for i in range(1, self.t1+1)]
        self.b_out_grad = sum(b_out_grad)/self.t1
        
        self.outer_grads = [self.W_out_grad, self.b_out_grad]
        
        get_a_J = self.net.get_a_jacobian
        
        if self.use_historical_W:
        
            self.a_Js = [get_a_J(update=False,
                         h=self.h_hist[-i],
                         h_prev=self.h_hist[-(i+1)],
                         W_rec=self.W_rec_hist[-i]) for i in range(1, self.T)]
        else:
            
            self.a_Js = [get_a_J(update=False,
                         h=self.h_hist[-i],
                         h_prev=self.h_hist[-(i+1)],
                         W_rec=self.net.W_rec) for i in range(1, self.T)] 
        
        n_h, n_in = self.net.n_hidden, self.net.n_in
        self.rec_grad = np.zeros((n_h, n_h+n_in+1))
        
        CA_list = []
        
        for i in range(1, self.t1+1):
            
            self.q = self.e_hist[-i].dot(self.net.W_out)
            
            for j in range(self.t2):
                
                self.J = np.eye(n_h)
                
                for k in range(j):
                    
                    self.J = self.J.dot(self.a_Js[-(i+k)])
                
                self.J = self.J.dot(np.diag(self.net.activation.f_prime(self.h_hist[-(i+j)])))
                
                self.pre_activity = np.concatenate([self.h_hist[-(i+j+1)], self.x_hist[-(i+j+1)], np.array([1])])
                CA = self.q.dot(self.J)
                CA_list.append(CA) 
                self.rec_grad += np.multiply.outer(CA, self.pre_activity)
        
        self.credit_assignment = sum(CA_list)/len(CA_list)
        
        grads = [self.rec_grad[:,:n_h], self.rec_grad[:,n_h:-1], self.rec_grad[:,-1]]
        grads += self.outer_grads
        
        self.update_monitors()
        
        return grads
    
    