#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 12:54:56 2018

@author: omarschall
"""

import numpy as np
import time
from copy import copy
from pdb import set_trace
from analysis_funcs import *

class Simulation:
    
    def __init__(self, net, learn_alg=None, optimizer=None, allowed_kwargs_=set(), **kwargs):
        
        '''
        Runs the network forward with inputs provided by data, a dict
        with entries data['train'] and data['test'], each of which
        is a dictionary with keys 'X' and 'Y' providing input data
        and labels, respectively, as numpy arrays with shape
        (time steps, units).
        
        ___Arguments___
        
        net                     Instance of class RNN or similar object
        
        learn_alg               Instance of the class Learning_Algorithm
                                (see learning_algorithms.py) used to train
                                the model if mode='train'.
                                
        optimizer               Instance of the class SGD or Adam (see
                                optimizers.py), used to take gradients from
                                learn_alg to 
                                
        comparison_alg          Instance of the class Learning_Algorithm
                                which does *not* train the model, but computes
                                gradients which are compared with learn_alg.
                                Default is none.
                                
        L2_reg                  Strength of L2 regularization used on non-bias
                                trainable parameters in the model. Default is 0.
                                
        monitors                List of strings dictating which variables should
                                be tracked while running. The dictionary self.mons
                                will be initialized with keys given by strings
                                in monitors.
                                
        update_interval         Number of time steps per update of the network's
                                parameters. Default is 1.
                                
        verbose                 Boolean dictating whether to report progress during
                                training. Default is True.
                                
        report_interval         Number of time steps per progress report. Default is
                                1/10 of the total time steps for 10 total updates.
                                
        check_accuracy          Boolean dictating whether to freeze parameters and
                                run in 'test' mode on every report. Default is False.
                                
        mode                    String that must be either 'train' or 'test' to
                                dictate which dataset to use and whether to update
                                parameters while running. Default is 'train'.
        '''


        allowed_kwargs = {'L2_reg', 'update_interval', 'sigma',
                          'verbose', 'report_interval', 'comparison_alg', 'mode',
                          'check_accuracy', 't_stop_SG_train', 't_stop_training',
                          'tau_avg', 'check_loss'}.union(allowed_kwargs_)
        for k in kwargs:
            if k not in allowed_kwargs:
                raise TypeError('Unexpected keyword argument '
                                'passed to Simulation.__init__: ' + str(k))
                
        #Store required args as part of simulation
        self.net       = net
        self.learn_alg = learn_alg
        self.optimizer = optimizer
        
        #Default simulation parameters
        self.sigma            = 0
        self.L2_reg           = 0
        self.update_interval  = 1
        
        #Overwrite defaults with any provided keyword args
        self.__dict__.update(kwargs)

    def run(self, data, mode='train', monitors=[], **kwargs):
        
        allowed_kwargs = {'verbose', 'report_interval', 'check_accuracy', 'check_loss'}
        for k in kwargs:
            if k not in allowed_kwargs:
                raise TypeError('Unexpected keyword argument '
                                'passed to Simulation.run: ' + str(k))
        
        net = self.net
        
        self.mode           = mode
        self.verbose        = True
        self.check_accuracy = False
        self.check_loss     = False
        
        #Make local copies of (meta)-data
        x_inputs             = data[self.mode]['X']
        y_labels             = data[self.mode]['Y']
        self.T               = x_inputs.shape[0]
        self.report_interval = max(self.T//10, 1)
        
        #Overwrite defaults with any provided keyword args
        self.__dict__.update(kwargs)

        #Initialize monitors
        self.mons = {}
        for mon in monitors:
            self.mons[mon] = []
        
        #Set a random initial state of the network
        net.reset_network()
        
        #Make local copies of (meta-)data
        x_inputs = data[self.mode]['X']
        y_labels = data[self.mode]['Y']
        self.T   = x_inputs.shape[0]
        
        #To avoid errors, initialize "previous"
        #inputs/labels as the first inputs/labels
        net.x_prev = x_inputs[0]
        net.y_prev = y_labels[0]
        
        #Track computation time
        self.t1 = time.time()
        
        for i_t in range(self.T):
            
            ### --- Run network forwards and get error --- ###
            
            self.forward_pass(x_inputs[i_t], y_labels[i_t])
            
            ### --- Update parameters if in 'train' mode --- ###

            if self.mode=='train':
                self.train_step(i_t)
            
            #Model-specific updates
            try:
                net.model_specific_updates()
            except AttributeError:
                pass
            
            if hasattr(self, 't_stop_SG_train'):
                if self.t_stop_SG_train==i_t:
                    self.learn_alg.optimizer.lr = 0
                    
            #Current inputs/labels become previous inputs/labels
            net.x_prev = np.copy(net.x)
            net.y_prev = np.copy(net.y)
            
            #Compute spectral radii if desired
            self.get_spectral_radii()
            
            #Monitor relevant variables
            self.update_monitors()
            
            #Make report if conditions are met
            if (i_t%self.report_interval)==0 and i_t>0 and self.verbose:
                self.report_progress(i_t, data)
                
            if hasattr(self, 't_stop_training') and self.mode=='train':
                if self.t_stop_training==i_t:
                    break
        
        #At end of run, convert monitor lists into numpy arrays
        self.monitors_to_arrays()

    def forward_pass(self, x, y):
        
        net = self.net
        
        net.x = x
        net.y = y
        
        net.next_state(net.x, sigma=self.sigma)
        net.z_out()
        
        net.y_hat  = net.output.f(net.z)
        net.loss_  = net.loss.f(net.z, net.y)
        net.e      = net.loss.f_prime(net.z, net.y)

    def train_step(self, i_t):
        
        net = self.net
        
        #Update internal learning algorithm parameters
        self.learn_alg.update_learning_vars()
        #Use learning algorithm to generate gradients
        self.grads = self.learn_alg()
        
        #If a comparison algorithm is provided, update
        #its internal parameters, generate gradients,
        #and measure their alignment with the main
        #algorithm's gradients.
        if hasattr(self, 'comparison_alg'):
            self.comparison_alg.update_learning_vars()
            self.grads_ = self.comparison_alg()
            G = zip(self.grads, self.grads_)
            try:
                self.alignment = [normalized_dot_product(g,g_) for g, g_ in G]
            except RuntimeWarning:
                self.alignment = [0]*5
            self.W_rec_alignment = self.alignment[0]
            self.W_in_alignment  = self.alignment[1]
            self.b_rec_alignment = self.alignment[2]
            self.W_out_alignment = self.alignment[3]
            self.b_out_alignment = self.alignment[4]
        
        #Add L2 regularization derivative to gradient
        if self.L2_reg>0:
            self.L2_regularization()
            
        #If on the update cycle (always true for update_inteval=1),
        #pass gradients to the optimizer and update parameters.
        if (i_t + 1)%self.update_interval==0:
            net.params = self.optimizer.get_update(net.params, self.grads)
            net.W_rec, net.W_in, net.b_rec, net.W_out, net.b_out = net.params
    
    def L2_regularization(self):
        
        for i_L2, W in zip(self.net.L2_indices, [self.net.params[i] for i in self.net.L2_indices]):
            self.grads[i_L2] += self.L2_reg*W
    
    def report_progress(self, i_t, data):
        
        t2 = time.time()
        
        progress = np.round((i_t/self.T)*100, 2)
        time_elapsed = np.round(t2 - self.t1, 1)
        
        summary = '\rProgress: {}% complete \nTime Elapsed: {}s \n'
        
        if 'loss_' in self.mons.keys():
            avg_loss = sum(self.mons['loss_'][-self.report_interval:])/self.report_interval
            loss = 'Average loss: {} \n'.format(avg_loss)
            summary += loss
            
        if self.check_accuracy or self.check_loss:
            test_sim = copy(self)
            test_sim.run(data, mode='test', monitors=['y_hat', 'loss_'], verbose=False)
            if self.check_accuracy:
                acc = classification_accuracy(data, test_sim.mons['y_hat'])
                accuracy = 'Test accuracy: {} \n'.format(acc)
                summary += accuracy
            if self.check_loss:
                test_loss = np.mean(test_sim.mons['loss_'])
                loss_summary = 'Test loss: {} \n'.format(test_loss)
                summary += loss_summary
            
        print(summary.format(progress, time_elapsed))
    
    def update_monitors(self):

        for key in self.mons.keys():
            for obj in [self, self.net, self.learn_alg, self.optimizer]:
                try:
                    self.mons[key].append(getattr(obj, key))
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
            
    def monitors_to_arrays(self):
        
        #After run is finished, turn monitors from lists into arrays.
        for key in self.mons.keys():
            if key=='a_s':
                self.mons[key] = np.concatenate(self.mons[key])
                continue
            try:
                self.mons[key] = np.array(self.mons[key])
            except ValueError:
                pass
            
    def get_spectral_radii(self):
        
        for key in self.mons.keys():
            if 'radius' in key:
                a = key.split('_')[0]
                for obj in [self, net, self.learn_alg]:
                    if hasattr(obj, a):
                        setattr(self, key, get_spectral_radius(getattr(obj, a)))
    