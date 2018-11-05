 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  7 17:20:39 2018

@author: omarschall
"""

import numpy as np
from utils import *
from optimizers import *
from analysis_funcs import *
import time
from copy import copy
from pdb import set_trace

class RNN:
    
    def __init__(self, W_in, W_rec, W_out, b_rec, b_out, activation, alpha, output, loss):
        '''
        Initializes a vanilla RNN object that follows the forward equation
        
        h_t = (1 - alpha)*h_{t-1} + W_rec * phi(h_{t-1}) + W_in * x_t + b_rec
        z_t = W_out * a_t + b_out
        
        with initial parameter values given by W_in, W_rec, W_out, b_rec, b_in
        and specified activation and loss functions, which must be function
        objects--see utils.py.
        
        ___Arguments___
        
        W_*                 Initial values of (in)put, (rec)urrent and (out)put
                            weights in the network.
                            
        b_*                 Initial values of (rec)urrent and (out)put biases.
        
        activation          Instance of function class (see utils.py) used for
                            calculating activations a from pre-activations h.
                            
        alpha               Ratio of time constant of integration to time constant
                            of leak.
                            
        output              Instance of function class used for calculating final
                            output from z.
                            
        loss                Instance of function class used for calculating loss
                            from z (must implicitly include output function, e.g.
                            softmax_cross_entropy if output is softmax).
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
        
        self.h = self.W_rec.dot(self.a) + self.W_in.dot(self.x) + self.b_rec
        self.a = (1 - self.alpha)*self.a + self.activation.f(self.h)
        
    def z_out(self):
        '''
        Updates the output of the RNN using the current activations
        '''
        
        self.z_prev = np.copy(self.z)
        self.z = self.W_out.dot(self.a) + self.b_out
            
    def get_a_jacobian(self, update=True, **kwargs):
        '''
        By default, it updates the Jacobian of the network,
        self.a_J, to the value based on the current parameter
        values and pre-activations. If update=False, then
        it does *not* update self.a_J, but rather returns
        the Jacobian calculated from current pre-activation
        values. If a keyword argument for 'h' or 'W_rec' is
        provided, these arguments are used instead of the
        network's current values.
        '''
        
        #Use kwargs instead of defaults if provided
        try:
            h = kwargs['h']
        except KeyError:
            h = np.copy(self.h)
        try:
            W_rec = kwargs['W_rec']
        except KeyError:
            W_rec = np.copy(self.W_rec)
        
        #Element-wise nonlinearity derivative
        D = self.activation.f_prime(h)
        a_J = np.diag(D).dot(W_rec) + (1 - self.alpha)*np.eye(self.n_hidden)
        
        if update:
            self.a_J = np.copy(a_J)
        else:
            return a_J
        
    def run(self, data, learn_alg=None, optimizer=None, **kwargs):
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
        #Store required args as part of network
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
        self.report_interval  = max(self.data['train']['X'].shape[0]//10, 1)
        
        #Overwrite defaults with any provided keyword args
        self.__dict__.update(kwargs)
        
        #Set a random initial state of the network
        self.reset_network()
        
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
        self.x_prev = x_inputs[0]
        self.y_prev = y_labels[0]
        
        #Track computation time
        self.t1 = time.time()
        
        for i_t in range(self.T):
            
            ### --- Run network forwards and get error --- ###
            
            self.x = x_inputs[i_t]
            self.y = y_labels[i_t]
            
            self.next_state(self.x)
            self.z_out()
            
            self.y_hat  = self.output.f(self.z)
            self.loss_  = self.loss.f(self.z, self.y)
            self.e      = self.loss.f_prime(self.z, self.y)

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
    
    
    
    
    
    
    
    
    
    
    