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
from utils import *
from state_space import State_Space_Analysis
import pickle
import os

class Simulation:
    """Simulates an RNN for a provided set of inputs and training procedures.

    By default, all variables are overwritten at each time step, but the user
    can specify which variables to track with 'monitors'.
    
    Has two types of attributes that are provided either upon init or
    when calling self.run. The distinction matters because init attributes
    should carry over into test runs, whereas run attributes will likely be
    different in train and test runs. Details given in __init__ and run
    docstrings.
    """
    
    def __init__(self, net, allowed_kwargs_=set(), **kwargs):
        """Initialzes a simulation.Simulation object by specifying the
        attributes that will apply to both train and test instances.
        
        Args:
            net (network.RNN): The specific RNN instance being simulated.
            allowed_kwargs_ (set): Custom allowed keyword args for development
                of subclasses of simulation that need additional specification.
            time_steps_per_trial (int): Number of time steps in a trial, if
                task has a trial structure. Leave empty if task runs
                continuously.
            trial_lr_mask (numpy array): Array of shape (time_steps_per_trial)
                that scales the loss at each time step within a trial.
            reset_sigma (float): Standard deviation of RNN initial state at
                start of each trial. Leave empty if RNN state should carry over
                from last state of previous trial.
            i_job (int): An integer indexing the job that this simulation
                corresponds to if submitting a batch job to cluster.
            save_dir (string): Path indicating where to save intermediate
                results, if desired.
        """

        allowed_kwargs = {'time_steps_per_trial',
                          'reset_sigma', 'trial_lr_mask',
                          'i_job', 'save_dir'}.union(allowed_kwargs_)
        for k in kwargs:
            if k not in allowed_kwargs:
                raise TypeError('Unexpected keyword argument '
                                'passed to Simulation.__init__: ' + str(k))
        self.net = net
        self.__dict__.update(kwargs)
        
        #Set to None all unspecified attributes
        for attr in allowed_kwargs:
            if not hasattr(self, attr):
                setattr(self, attr, None)

    def run(self, data, mode='train', monitors=[], **kwargs):
        """Runs the network forward as many time steps as given by data.
        
        Can be run in either 'train' or 'test' mode, which specifies whether
        the network parameters are updated or not. In 'train' case, a
        learning algorithm (learn_alg) and optimizer (optimizer) must be
        specified to calculate gradients and apply them to the network,
        respectively.
        
        Args:
            data (dict): A dict containing two keys, 'train' and 'test',
                each of which points to a dict containing keys 'X' and 'Y',
                providing numpy arrays of inputs and labels, respectively,
                of shape (T, n_in) or (T, n_out).
            mode (string): A string that must be either 'train' or 'test'
                which indicates which dict in data to use and whether to
                update network parameters while running.
            monitors (list): A list of strings, such that if a string matches
                an attribute of any relevant object in the simluation,
                including the simulation itself, the network, the optimizer,
                or the learning algorithm, that attribute's value is stored
                at each time step. If there is a '-' (hyphen) between the name
                and either 'radius' or 'norm', then the spectral radius or
                norm, respectively, of that object is stored instead.
            learn_alg (learning_algorithms.Learning_Algorithm): The instance
                of a learning algorithm used to calculate the gradients.
            optimizer (optimizers.Optimizer): The instance of an optimizer
                used to update the network using gradients computed by
                learn_alg.
            update_interval (int): Number of time steps between each parameter
                update.
            a_initial (numpy array): An array of shape (net.n_hidden) that
                specifies the initial state of the network when running. If
                not specified, the default initialization practice is inherited
                from the RNN.
            sigma (float): Specifies standard deviation of white noise to add
                to the network pre-activations at each time step.
            comp_algs (list): A list of instances of Learning_Algorithm
                specified to passively run in parallel with learn_alg to enable
                comparisons between the algorithms.
            verbose (bool): Boolean that indicates whether to print progress
                reports during the simulation.
            report_interval (int): Number of time steps between reports, if
                verbose. Default is 1/10 the number of total time steps, for 10
                total progress reports.
            check_accuracy (bool): Boolean that indicates whether to run a test
                simulation during each progress report to report classification
                accuracy, as defined by the fraction of test time steps where
                the argmax of the outputs matches that of the labels.
            check_loss (bool): Same as check_accuracy but with test loss. If
                both are False, no test simulation is run.
            save_model_interval (int): Number of time steps between running
                test simulations and saving the model if it has the lowest
                yet validation loss.
        
        """
        
        allowed_kwargs = {'learn_alg', 'optimizer', 'a_initial', 'sigma',
                          'update_interval', 'comp_algs', 'verbose',
                          'report_interval', 'check_accuracy', 'check_loss',
                          'save_model_interval'}
        for k in kwargs:
            if k not in allowed_kwargs:
                raise TypeError('Unexpected keyword argument '
                                'passed to Simulation.run: ' + str(k))
        
        ### --- Set new object attributes for run --- ###
        
        #Create new pointers for conveneince
        self.mode = mode
        self.monitors = monitors
        self.x_inputs = data[mode]['X']
        self.y_labels = data[mode]['Y']
        self.T = self.x_inputs.shape[0]
        
        #Set defaults
        self.verbose = True
        self.check_accuracy = False
        self.check_loss = False
        self.comp_algs = []
        self.report_interval = max(self.T//10, 1)
        self.update_interval = 1
        self.sigma = 0
        
        #Overwrite defaults with any provided keyword args
        self.__dict__.update(kwargs)
        
        #Set to None all unspecified attributes
        for attr in allowed_kwargs:
            if not hasattr(self, attr):
                setattr(self, attr, None)

        ### --- Pre-run housekeeping --- ###

        self.initialize_run()
        
        for i_t in range(self.T):
            
            ### --- Reset model if there is a trial structure --- ###
            
            if self.time_steps_per_trial is not None:
                self.trial_structure()
            
            ### --- Run network forwards and get error --- ###
            
            self.forward_pass(self.x_inputs[i_t],
                              self.y_labels[i_t])
            
            ### --- Update parameters if in 'train' mode --- ###

            if self.mode=='train':
                self.train_step(i_t)
                
            ### --- Clean up --- ###
            
            self.end_time_step(i_t)
        
        #At end of run, convert monitor lists into numpy arrays
        self.monitors_to_arrays()

    def initialize_run(self):
        
        #Initial best validation loss is infinite
        self.best_val_loss = np.inf
        
        #Initialize monitors
        self.mons = {}
        for mon in self.monitors:
            self.mons[mon] = []
        self.objs = [self, self.net, self.learn_alg, self.optimizer]
        self.objs += self.comp_algs
        
        #Set a random initial state of the network
        if self.a_initial is not None:
            self.net.reset_network(a=self.a_initial)
        else:
            self.net.reset_network()
        
        #To avoid errors, initialize "previous"
        #inputs/labels as the first inputs/labels
        self.net.x_prev = self.x_inputs[0]
        self.net.y_prev = self.y_labels[0]
        
        #Track computation time
        self.t1 = time.time()

    def trial_structure(self, i_t):
        
        self.i_t_trial = i_t%self.time_steps_per_trial
        if self.i_t_trial == 0:
            self.i_trial = i_t//self.time_steps_per_trial
            if self.reset_sigma is not None:
                self.net.reset_network(sigma=self.reset_sigma)
                self.learn_alg.reset_learning()
        
    def forward_pass(self, x, y):
        
        #Pointer for convenience
        net = self.net
        
        #Pass data to network
        net.x = x
        net.y = y
        
        #Run network forwards and get predictions
        net.next_state(net.x, sigma=self.sigma)
        net.z_out()
        
        #Compare outputs with labels, get immediate loss and errors
        net.y_hat = net.output.f(net.z)
        net.loss_  = net.loss.f(net.z, net.y)
        net.e      = net.loss.f_prime(net.z, net.y)
        
        #Re-scale losses and errors if trial structure is provided
        if self.trial_lr_mask is not None:
            self.loss_ *= self.trial_lr_mask[self.i_t_trial]
            self.e *= self.trial_lr_mask[self.i_t_trial]            

    def train_step(self, i_t):
        
        ### --- Calculate gradients --- ###
        
        #Pointer for convenience
        net = self.net
        
        #Update learn_alg variables and get gradients
        self.learn_alg.update_learning_vars()
        self.grads_list = self.learn_alg()
        
        ### --- Calculate gradients for comparison algorithms --- ###
        
        self.rec_grads_dict = {'learn_alg': self.learn_alg.rec_grads}
        for i_comp_alg, comp_alg in enumerate(self.comp_algs):
            comp_alg.update_learning_vars()
            _ = comp_alg()
            key = 'comp_alg_{}'.format(i_comp_alg)
            self.rec_grads_dict[key] = comp_alg.rec_grads
        if 'alignment_matrix' in self.mons.keys():
            n_algs = 1 + len(self.comp_algs)
            self.alignment_matrix = np.zeros((n_algs, n_algs))
            for i in range(n_algs):
                for j in range(n_algs):
                    g1 = self.rec_grads_dict.values()[i]
                    g2 = self.rec_grads_dict.values()[j]
                    self.alignment_matrix[i,j] = normalized_dot_product(g1, g2)
            
        ### --- Pass gradients to optimizer --- ###
        
        #Only update on schedule (default update_interval=1)
        if i_t%self.update_interval==0:
            #Get updated parameters
            net.params = self.optimizer.get_update(net.params,
                                                   self.grads_list)
            net.W_rec, net.W_in, net.b_rec, net.W_out, net.b_out = net.params
    
    def end_time_step(self, i_t):
        
        #Current inputs/labels become previous inputs/labels
        self.net.x_prev = np.copy(self.net.x)
        self.net.y_prev = np.copy(self.net.y)
        
        #Compute spectral radii if desired
        self.get_radii_and_norms()
        
        #Monitor relevant variables
        self.update_monitors()
        
        #Evaluate model and save if performance is best
        if self.save_model_interval is not None and mode=='train':
            if i_t%self.save_model_interval==0:
                self.save_best_model(data)
        
        #Make report if conditions are met
        if i_t%self.report_interval==0 and i_t>0 and self.verbose:
            self.report_progress(i_t, data)
    
    def report_progress(self, i_t, data):
        
        t2 = time.time()
        
        progress = np.round((i_t/self.T)*100, 2)
        time_elapsed = np.round(t2 - self.t1, 1)
        
        summary = '\rProgress: {}% complete \nTime Elapsed: {}s \n'
        
        if 'loss_' in self.mons.keys():
            interval = self.report_interval
            avg_loss = sum(self.mons['loss_'][-interval:])/interval
            loss = 'Average loss: {} \n'.format(avg_loss)
            summary += loss
            
        if self.check_accuracy or self.check_loss:
            test_sim = self.get_test_sim()
            test_sim.run(data, mode='test',
                         monitors=['y_hat', 'loss_'],
                         verbose=False)
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
            for obj in self.objs:
                try:
                    self.mons[key].append(getattr(obj, key))
                except AttributeError:
                    pass
            
    def monitors_to_arrays(self):
        
        for key in self.mons.keys():
            try:
                self.mons[key] = np.array(self.mons[key])
            except ValueError:
                pass
            
    def get_radii_and_norms(self):
        
        for feature, func in zip(['radius', 'norm'], [get_spectral_radius, norm]):
            for key in self.mons.keys():
                if feature in key:
                    a = key.split('-')[0]
                    for obj in [self, self.net, self.learn_alg]:
                        if hasattr(obj, a):
                            setattr(self, key, func(getattr(obj, a)))
                        
    def save_best_model(self, data):
        
        val_sim = self.get_test_sim()
        val_sim.run(data, mode='test',
                    monitors=['y_hat', 'loss_'],
                    verbose=False)
        val_loss = np.mean(val_sim.mons['loss_'])
        
        if val_loss < self.best_val_loss:
            self.best_net = copy(self.net)
            self.best_val_loss = val_loss
            
    def get_test_sim(self):
        
        sim = Simulation(self.net,
                         time_steps_per_trial=self.time_steps_per_trial,
                         reset_sigma=self.reset_sigma,
                         i_job=self.i_job,
                         save_dir=self.save_dir)
        return sim
            

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        