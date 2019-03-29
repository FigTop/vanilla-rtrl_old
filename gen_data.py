#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 16:30:03 2018

@author: omarschall
"""

import numpy as np
import matplotlib.pyplot as plt
from copy import copy
from pdb import set_trace

class Task:
    
    def __init__(self, n_in, n_out):
        
        self.n_in = n_in
        self.n_out = n_out
        
    def plot_filtered_signals(self, signals,
                              plot_loss_benchmarks=True,
                              filter_size=100,
                              **kwargs):
    
        for signal in signals:
            smoothed_signal = np.convolve(signal, np.ones(filter_size)/filter_size, mode='valid')
            plt.plot(smoothed_signal)
        
        if plot_loss_benchmarks:
            try:
                self.plot_loss_benchmarks()
            except AttributeError:
                pass
        
        try:
            plt.ylim(kwargs['y_lim'])
        except KeyError:
            pass
        
        plt.xlabel('Time')
    
    def gen_data(self, N_train, N_test):
        
        data = {'train': {}, 'test': {}}
        
        data['train']['X'], data['train']['Y'] = self.gen_dataset(N_train)
        data['test']['X'], data['test']['Y'] = self.gen_dataset(N_test)
        
        self.data = data
        
        return data
         
class Coin_Task(Task):
    
    def __init__(self, n1, n2, one_hot=True, deterministic=False, tau_task=1):
        
        super().__init__(2, 2)
        
        #Dependencies in coin task
        self.n1 = n1
        self.n2 = n2
        self.tau_task = tau_task
        
        #Use one hot representation of coin flips or not
        self.one_hot = one_hot
        #Use coin flip outputs or deterministic probabilities as labels
        self.deterministic = deterministic
        
    def gen_dataset(self, N):
        
        if self.one_hot:
        
            X = []
            Y = []
            
            N = N//self.tau_task
            
            for i in range(N):
                
                x = np.random.binomial(1, 0.5)
                X.append(np.array([x, 1-x]))
                
                p = 0.5
                try:
                    p += X[-self.n1][0]*0.5
                except IndexError:
                    pass
                try:
                    p -= X[-self.n2][0]*0.25
                except IndexError:
                    pass
                
                if not self.deterministic:
                    y = np.random.binomial(1, p)
                else:
                    y = p
                Y.append(np.array([y, 1-y]))
        
            X = np.array(X)
            Y = np.array(Y)
            
            X = np.tile(X, self.tau_task).reshape((self.tau_task*N, 2))
            Y = np.tile(Y, self.tau_task).reshape((self.tau_task*N, 2))
        
        else:
            
            X = np.random.binomial(1, 0.5, N).reshape((-1, 1))
            Y = np.zeros_like(X)
            for i in range(N):
                p = 0.5
                if X[i-self.n1,0]==1:
                    p += 0.5
                if X[i-self.n2,0]==1:
                    p -= 0.25
                Y[i,0] = np.random.binomial(1, p)
            
        return X, Y
    
    def plot_loss_benchmarks(self):
        
        plt.axhline(y=0.66, color='r', linestyle='--')
        plt.axhline(y=0.52, color='b', linestyle='--')
        plt.axhline(y=0.45, color='g', linestyle='--')

class Copy_Task(Task):
    
    def __init__(self, n_symbols, T):
        
        super().__init__(n_symbols + 1, n_symbols + 1)
        
        self.n_symbols = n_symbols
        self.T = T

    def gen_dataset(self, N):
        
        n_sequences = N//(2*self.T)
        
        I = np.eye(self.n_in)
    
        X = np.zeros((1, self.n_in))
        Y = np.zeros((1, self.n_in))
        
        for i in range(n_sequences):
            
            seq = I[np.random.randint(0, self.n_symbols, size=self.T)]
            cue = np.tile(I[-1], (self.T, 1))
            X = np.concatenate([X, seq, cue])
            Y = np.concatenate([Y, cue, seq])
            
        return X, Y
    
class Mimic_RNN(Task):
    
    def __init__(self, rnn, p_input):
        
        super().__init__(rnn.n_in, rnn.n_out)
        
        self.rnn = rnn
        self.p_input = p_input
        
    def gen_dataset(self, N):
        
        X = np.random.binomial(1, self.p_input, (N, self.n_in))
        
        Y = []
        self.rnn.reset_network()
        for i in range(N):
            self.rnn.next_state(X[i])
            self.rnn.z_out()
            Y.append(self.rnn.output.f(self.rnn.z))
            
        return X, np.array(Y)
            
class Sine_Wave(Task):
    
    def __init__(self, p_transition, frequencies, never_off=False, **kwargs):
        
        allowed_kwargs = {'p_frequencies', 'amplitude', 'method'}
        for k in kwargs:
            if k not in allowed_kwargs:
                raise TypeError('Unexpected keyword argument '
                                'passed to Sine_Wave.__init__: ' + str(k))
        
        super().__init__(2, 2)
        
        self.p_transition = p_transition
        self.method = 'random'
        self.amplitude = 0.1
        self.frequencies = frequencies
        self.p_frequencies = np.ones_like(frequencies)/len(frequencies)
        self.never_off = never_off
        self.__dict__.update(kwargs)
                 
    def gen_dataset(self, N):
        
        X = np.zeros((N, 2))
        Y = np.zeros((N, 2))
        
        self.switch_cond = False
        
        active = False
        t = 0
        X[0,0] = 1
        for i in range(1, N):
            
            if self.method=='regular':
                if i%int(1/self.p_transition)==0:
                    self.switch_cond = True
            elif self.method=='random':
                if np.random.rand()<self.p_transition:
                    self.switch_cond = True
            
            if self.switch_cond:  
                
                t = 0
                
                if active and not self.never_off:
                    X[i,0] = 1
                    X[i,1] = 0
                    Y[i,:] = 0
                
                if not active or self.never_off:
                    X[i,0] = np.random.choice(self.frequencies, p=self.p_frequencies)
                    X[i,1] = 1
                    Y[i,0] = self.amplitude*np.cos(2*np.pi*X[i,0]*t)
                    Y[i,1] = self.amplitude*np.sin(2*np.pi*X[i,0]*t)
                
                active = not active
                
            else:
                
                t+=1
                X[i,:] = X[i-1,:]
                Y[i,0] = self.amplitude*np.cos(2*np.pi*X[i,0]*t)*(active or self.never_off)
                Y[i,1] = self.amplitude*np.sin(2*np.pi*X[i,0]*t)*(active or self.never_off)
                
            self.switch_cond = False
                
        X[:,0] = -np.log(X[:,0])
                
        return X, Y
        
class Sensorimotor_Mapping(Task):
    
    def __init__(self, t_stim=1, stim_duration=3,
                       t_report=20, report_duration=3):
        
        super().__init__(2, 2)
        
        self.t_stim = t_stim
        self.stim_duration = stim_duration
        self.t_report = t_report
        self.report_duration = report_duration
        self.time_steps_per_trial = t_report + report_duration
        
        #Make mask for preferential learning within task
        self.trial_lr_mask = np.ones(self.time_steps_per_trial)*0.1
        self.trial_lr_mask[self.t_report:self.t_report+self.report_duration] = 1
        
    def gen_dataset(self, N):
        
        X = []
        Y = []
        
        for i in range(N//self.time_steps_per_trial):
            
            x = np.zeros((self.time_steps_per_trial, 2))
            y = np.ones_like(x)*0.5
            
            LR = 2*np.random.binomial(1, 0.5) - 1
            x[self.t_stim:self.t_stim+self.stim_duration, 0] = LR
            x[self.t_report:self.t_report+self.report_duration, 1] = 1
            y[self.t_report:self.t_report+self.report_duration, 0] = 0.5*(LR + 1)
            y[self.t_report:self.t_report+self.report_duration, 1] = 1 - 0.5*(LR + 1)
        
            X.append(x)
            Y.append(y)
            
        X = np.concatenate(X, axis=0)
        Y = np.concatenate(Y, axis=0)
        
        return X, Y
        
        
        
        
        
        
        
        
        
        
        