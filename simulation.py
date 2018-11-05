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

class Simulation():
    
    def __init__(self, net, learn_alg=None, optimizer=None, **kwargs):
        
        '''
        Runs the network forward with inputs provided by data, a dict
        with entries data['train'] and data['test'], each of which
        is a dictionary with keys 'X' and 'Y' providing input data
        and labels, respectively, as numpy arrays with shape
        (time steps, units).
        
        ___Arguments___
        
        data                    Dict providing input and label data
        
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
                                
        l2_reg                  Strength of L2 regularization used on non-bias
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


        allowed_kwargs = {'l2_reg', 'monitors', 'update_interval',
                          'verbose', 'report_interval', 'comparison_alg', 'mode',
                          'check_accuracy', 't_stop_SG_train', 't_stop_training'}
        for k in kwargs:
            if k not in allowed_kwargs:
                raise TypeError('Unexpected keyword argument '
                                'passed to RNN.run: ' + str(k))
                
        #Store required args as part of simulation
        self.net       = net
        self.learn_alg = learn_alg
        self.optimizer = optimizer
        
        #Default simulation parameters
        self.l2_reg           = 0
        self.monitors         = []
        self.update_interval  = 1
        self.verbose          = True
        
        
        #Overwrite defaults with any provided keyword args
        self.__dict__.update(kwargs)

    def run(self, data, mode='train', monitors=[]):
        
        self.mode = mode
        
        #Make local copies of (meta)-data
        x_inputs = data[self.mode]['X']
        y_labels = data[self.mode]['Y']
        self.T   = x_inputs.shape[0]
        self.report_interval  = max(self.T//10, 1)
        self.check_accuracy   = False
        
        #Initialize monitors
        self.mons = {}
        for mon in self.monitors:
            self.mons[mon] = []
        
        #Set a random initial state of the network
        self.net.reset_network()
        
        #Make local copies of (meta)-data
        x_inputs = data[self.mode]['X']
        y_labels = data[self.mode]['Y']
        self.T   = x_inputs.shape[0]
        
        #Initialize monitors
        self.mons = {}
        for mon in self.monitors:
            self.mons[mon] = []
        
        #To avoid errors, intialize "previous"
        #inputs/labels as the first inputs/labels
        self.net.x_prev = x_inputs[0]
        self.net.y_prev = y_labels[0]
        
        #Track computation time
        self.t1 = time.time()
        
        for i_t in range(self.T):
            
            ### --- Run network forwards and get error --- ###
            
            self.net.x = x_inputs[i_t]
            self.net.y = y_labels[i_t]
            
            self.net.next_state(self.net.x)
            self.net.z_out()
            
            self.net.y_hat  = self.net.output.f(self.net.z)
            self.net.loss_  = self.net.loss.f(self.net.z, self.net.y)
            self.net.e      = self.net.loss.f_prime(self.net.z, self.net.y)

            ### --- Update parameters if in 'train' mode --- ###

            if self.mode=='train':
                
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
                for i_l2, W in zip([0, 1, 3], [self.W_rec, self.W_in, self.W_out]):
                    self.grads[i_l2] += self.l2_reg*W
                    
                #If on the update cycle (always true for update_inteval=1),
                #pass gradients to the optimizer and update parameters.
                if (i_t + 1)%self.update_interval==0:
                    self.params = self.optimizer.get_update(self.params, self.grads)
                    self.W_rec, self.W_in, self.b_rec, self.W_out, self.b_out = self.params
                
                        
            if hasattr(self, 't_stop_SG_train'):
                if self.t_stop_SG_train==i_t:
                    self.learn_alg.optimizer.lr = 0
                    
            #Current inputs/labels become previous inputs/labels
            self.x_prev = np.copy(self.x)
            self.y_prev = np.copy(self.y)
            
            #Compute spectral radii if desired
            if 'W_radius' in self.mons.keys():
                self.W_radius = get_spectral_radius(self.W_rec)
            if 'A_radius' in self.mons.keys():
                self.A_radius = get_spectral_radius(learn_alg.A)
            
            #Monitor relevant variables
            self.update_monitors()
            
            #Make report if conditions are met
            if (i_t%self.report_interval)==0 and i_t>0 and self.verbose:
                self.report_progress(i_t)
                
            if hasattr(self, 't_stop_training') and self.mode=='train':
                if self.t_stop_training==i_t:
                    break
        
        #At end of run, convert monitor lists into numpy arrays
        self.monitors_to_arrays()
                
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
            
    def monitors_to_arrays(self):
        
        #After run is finished, turn monitors from lists into arrays.
        objs = [self]
        if self.learn_alg is not None:
            objs += [self.learn_alg]
        if hasattr(self, 'comparison_alg'):
            objs += [self.comparison_alg]
            
        for obj in objs:
            for key in obj.mons.keys():
                try:
                    obj.mons[key] = np.array(obj.mons[key])
                except ValueError:
                    pass
    