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
    """Parent class for all tasks."""

    def __init__(self, n_in, n_out):
        """Initializes a Task with the number of input and output dimensions

        Arguments:
            n_in (int): Number of input dimensions.
            n_out (int): Number of output dimensions."""

        self.n_in = n_in
        self.n_out = n_out

    def gen_data(self, N_train, N_test):
        """Generates a data dict with a given number of train and test examples.

        Arguments:
            N_train (int): number of training examples
            N_test (int): number of testing examples
        Returns:
            data (dict): Dictionary pointing to 2 sub-dictionaries 'train'
                and 'test', each of which has keys 'X' and 'Y' for inputs
                and labels, respectively."""

        data = {'train': {}, 'test': {}}

        data['train']['X'], data['train']['Y'] = self.gen_dataset(N_train)
        data['test']['X'], data['test']['Y'] = self.gen_dataset(N_test)

        return data

    def gen_dataset(self, N):
        """Function specific to each class, randomly generates a dictionary of
        inputs and labels.

        Arguments:
            N (int): number of examples
        Returns:
            dataset (dict): Dictionary with keys 'X' and 'Y' pointing to inputs
                and labels, respectively."""

        pass

class Add_Task(Task):
    """Class for the 'Add Task', an input-label mapping with i.i.d. Bernoulli
    inputs (p=0.5) and labels depending additively on previous inputs at
    t_1 and t_2 time steps ago:

    y(t) = 0.5 + 0.5 * x(t - t_1) - 0.25 * x(t - t_2)           (1)

    as inspired by Pitis 2016
    (https://r2rt.com/recurrent-neural-networks-in-tensorflow-i.html).

    The inputs and outputs each have a redundant dimension representing the
    complement of the outcome (i.e. x_1 = 1 - x_0), because keeping all
    dimensions above 1 makes python broadcasting rules easier."""

    def __init__(self, t_1, t_2, deterministic=False, tau_task=1):
        """Initializes an instance of this task by specifying the temporal
        distance of the dependencies, whether to use deterministic labels, and
        the timescale of the changes.

        Args:
            t_1 (int): Number of time steps for first dependency
            t_2 (int): Number of time steps for second dependency
            deterministic (bool): Indicates whether to take the labels as
                the exact numbers in Eq. (1) OR to use those numbers as
                probabilities in Bernoulli outcomes.
            tau_task (int): Factor by which we temporally 'stretch' the task. For
                example, if tau_task = 3, each input (and label) is repeated for
                3 time steps before being replaced by a new random sample."""

        #Initialize a parent Task object with 2 input and 2 output dimensions.
        super().__init__(2, 2)

        #Dependencies in coin task
        self.t_1 = t_1
        self.t_2 = t_2
        self.tau_task = tau_task

        #Use coin flip outputs or deterministic probabilities as labels
        self.deterministic = deterministic

    def gen_dataset(self, N):
        """Generates a dataset according to Eq. (1)."""

        #Generate random bernoulli inputs and labels according to Eq. (1).
        N = N // self.tau_task
        x = np.random.binomial(1, 0.5, N)
        y = 0.5 + 0.5 * np.roll(x, self.t_1) - 0.25 * np.roll(x, self.t_2)
        if not self.deterministic:
            y = np.random.binomial(1, y, N)
        X = np.array([x, 1 - x]).T
        Y = np.array([y, 1 - y]).T

        #Temporally stretch according to the desire timescale of change.
        X = np.tile(X, self.tau_task).reshape((self.tau_task*N, 2))
        Y = np.tile(Y, self.tau_task).reshape((self.tau_task*N, 2))

        return X, Y

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

    def __init__(self, rnn, p_input, tau_task=1):

        super().__init__(rnn.n_in, rnn.n_out)

        self.rnn = rnn
        self.p_input = p_input
        self.tau_task = tau_task

    def gen_dataset(self, N):


        N_tau = N // self.tau_task
        X = []
        for i in range(N_tau):
            x = np.random.binomial(1, self.p_input, self.n_in)
            X.append(x)
        X = np.tile(X, self.tau_task).reshape((self.tau_task*N_tau, self.n_in))

        Y = []
        self.rnn.reset_network()
        for i in range(len(X)):
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
        if self.method == 'regular':
            self.time_steps_per_trial = int(1/self.p_transition)
            self.trial_lr_mask = np.ones(self.time_steps_per_trial)

    def gen_dataset(self, N):

        X = np.zeros((N, 2))
        Y = np.zeros((N, 2))

        self.switch_cond = False

        active = False
        t = 0
        X[0,0] = 1
        for i in range(1, N):

            if self.method=='regular':
                if i%self.time_steps_per_trial==0:
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

class Repeat_Sequence(Task):

    def __init__(self, n_symbols, T_sequence, T_delay):

        super().__init__(n_symbols, n_symbols)

class Flip_Flop_Task(Task):
    """Generates data for the N-bit flip-flop task."""

    def __init__(self, n_bit, p_flip):

        super().__init__(self, n_bit, n_bit)

        self.p_flip = p_flip

    def gen_dataset(self, N):

        pass








