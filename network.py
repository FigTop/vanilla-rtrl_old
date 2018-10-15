#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  7 17:20:39 2018

@author: omarschall
"""

import numpy as np
from utils import *
from optimizers import *
from analysis_funcs import classification_accuracy, normalized_dot_product
import time
from copy import copy

class RNN:
    
    def __init__(self, W_in, W_rec, W_out, b_rec, b_out, activation, alpha, output, loss):
        '''
        Initializes a vanilla RNN object that follows the forward equation
        
        h_t = (1 - alpha)*h_{t-1} + W_rec * phi(h_{t-1}) + W_in * x_t + b_rec
        z_t = W_out * a_t + b_out
        
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
            
        try:
            W_rec = kwargs['W_rec']
        except KeyError:
            W_rec = np.copy(self.W_rec)
        
        q1 = self.activation.f_prime(h)
        
        if self.alpha!=1:
            q2 = self.activation.f_prime(h_prev)
            a_J = np.diag(q1).dot(W_rec + np.diag((1-self.alpha)/(q2)))
        else:
            a_J = np.diag(q1).dot(W_rec)
        
        if update:
            self.a_J = np.copy(a_J)
        else:
            return a_J
        
    def run(self, data, learn_alg=None, optimizer=None, **kwargs):
        
        allowed_kwargs = {'l2_reg', 'monitors', 'update_interval',
                          'verbose', 'report_interval', 'comparison_alg', 'mode',
                          'check_accuracy'}
        for k in kwargs:
            if k not in allowed_kwargs:
                raise TypeError('Unexpected keyword argument '
                                'passed to RNN.run: ' + str(k))
        self.data      = data
        self.learn_alg = learn_alg
        self.optimizer = optimizer
        
        #Default run parameters
        self.mode             = 'train'
        self.check_accuracy   = False
        self.l2_reg           = 0
        self.monitors         = []
        self.update_interval  = 1
        self.verbose          = True
        self.report_interval  = len(self.data['train']['X'])//10
        
        self.__dict__.update(kwargs)
        self.reset_network()
        
        x_inputs = data[self.mode]['X']
        y_labels = data[self.mode]['Y']
        self.T   = len(x_inputs)
        
        #Initialize monitors
        self.mons = {}
        for mon in self.monitors:
            self.mons[mon] = []
        
        self.x_prev = x_inputs[0]
        self.y_prev = y_labels[0]
        
        self.t1 = time.time()
        
        for i_t in range(self.T):
            
            self.x = x_inputs[i_t]
            self.y = y_labels[i_t]
            
            self.next_state(self.x)
            self.z_out()
            
            self.y_hat  = self.output.f(self.z)
            self.loss_  = self.loss.f(self.z, self.y)
            self.e = self.loss.f_prime(self.z, self.y)

            if self.mode=='train':
                
                self.learn_alg.update_learning_vars()
                self.grads = self.learn_alg()
                
                if hasattr(self, 'comparison_alg'):
                    self.comparison_alg.update_learning_vars()
                    self.grads_ = self.comparison_alg()
                    G = zip(self.grads, self.grads_)
                    self.alignment = [normalized_dot_product(g,g_) for g, g_ in G]
                
                #L2 Reg
                for i_l2, W in zip([0, 1, 3], [self.W_rec, self.W_in, self.W_out]):
                    self.grads[i_l2] += self.l2_reg*W
                
                if (i_t + 1)%self.update_interval==0:
                    self.params = self.optimizer.get_update(self.params, self.grads)
                    self.W_rec, self.W_in, self.b_rec, self.W_out, self.b_out = self.params
                    
            
            self.x_prev = np.copy(self.x)
            self.y_prev = np.copy(self.y)
            
            self.update_monitors()
                    
            if (i_t%self.report_interval)==0 and i_t>0 and self.verbose:
                
                self.report_progress(i_t)
                
    def report_progress(self, i_t):
        
        t2 = time.time()
        
        progress = np.round((i_t/self.T)*100, 2)
        time_elapsed = np.round(t2 - self.t1, 1)
        
        summary = '\rProgress: {}% complete \nTime Elapsed: {}s \n'
        
        if 'loss_' in self.mons.keys():
            avg_loss = sum(self.mons['loss_'][-self.report_interval:])/self.report_interval
            loss = 'Average loss: {} \n'.format(avg_loss)
            summary += loss
            
        if self.check_accuracy:
            self.rnn_copy = copy(self)
            self.rnn_copy.run(self.data, mode='test', monitors=['y_hat'], verbose=False)
            acc = classification_accuracy(self.data, self.rnn_copy.mons['y_hat'])
            accuracy = 'Test accuracy: {} \n'.format(acc)
            summary += accuracy
            
        print(summary.format(progress, time_elapsed))
    
    def update_monitors(self):

        for key in self.mons.keys():
            try:
                self.mons[key].append(getattr(self, key))
            except AttributeError:
                pass
        
    def truncate_history(self, T):
        '''
        Delete all history of an RNN past a certain point
        (useful if learning blows up and we want to see what hapepend before)
        '''
        
        objs = [self, self.learn_alg]
        if hasattr(self, 'comparison_alg'):
            objs += self.comparison_alg
            
        for obj in objs:
            if hasattr(obj, 'mons'):
                for key in obj.mons.keys():
                    obj.mons[key] = obj.mons[key][:T]
            
            
    
    
    
    
    
    
    
    
    
    
    