#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 16:30:03 2018

@author: omarschall
"""

import numpy as np

class Task:
    """Parent class for all tasks."""

    def __init__(self, n_in, n_out):
        """Initializes a Task with the number of input and output dimensions

        Args:
            n_in (int): Number of input dimensions.
            n_out (int): Number of output dimensions."""

        self.n_in = n_in
        self.n_out = n_out

    def gen_data(self, N_train, N_test):
        """Generates a data dict with a given number of train and test examples.

        Args:
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

        Args:
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
            tau_task (int): Factor by which we temporally 'stretch' the task.
                For example, if tau_task = 3, each input (and label) is repeated
                for 3 time steps before being replaced by a new random
                sample."""

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

class Mimic_RNN(Task):
    """Class for the 'Mimic Task,' where the inputs are random i.i.d. Bernoulli
    and the labels are the outputs of a fixed 'target' RNN that is fed these
    inputs."""

    def __init__(self, rnn, p_input, tau_task=1):
        """Initializes the task with a target RNN (instance of network.RNN),
        the probability of the Bernoulli inputs, and a time constant of change.

        Args:
            rnn (network.RNN instance): The target RNN
            p_input (float): The probability of any input having value 1
            tau_task (int): The temporal stretching factor for the inputs, see
                tau_task in Add_Task."""

        #Initialize as Task object with dims inherited from the target RNN.
        super().__init__(rnn.n_in, rnn.n_out)

        self.rnn = rnn
        self.p_input = p_input
        self.tau_task = tau_task

    def gen_dataset(self, N):
        """Generates a dataset by first generating inputs randomly by the
        binomial distribution and temporally stretching them by tau_task,
        then feeding these inputs to the target RNN."""

        #Generate inputs
        N = N // self.tau_task
        X = []
        for i in range(N):
            x = np.random.binomial(1, self.p_input, self.n_in)
            X.append(x)
        X = np.tile(X, self.tau_task).reshape((self.tau_task*N, self.n_in))

        #Get RNN responses
        Y = []
        self.rnn.reset_network(h=np.zeros(self.rnn.n_h))
        for i in range(len(X)):
            self.rnn.next_state(X[i])
            self.rnn.z_out()
            Y.append(self.rnn.output.f(self.rnn.z))

        return X, np.array(Y)

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

class Flip_Flop_Task(Task):
    """Generates data for the N-bit flip-flop task."""

    def __init__(self, n_bit, p_flip):

        super().__init__(self, n_bit, n_bit)

        self.p_flip = p_flip

    def gen_dataset(self, N):

        pass








