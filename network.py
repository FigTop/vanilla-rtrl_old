#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  7 17:20:39 2018

@author: omarschall
"""

import numpy as np
from utils import *
from functions import *

class RNN:
    """Generic class for any recurrent network.

    Attributes:
        n_in (int): Number of input dimensions
        n_state (int): Number of hidden units
        n_out (int): Number of output dimensions
        W_out (numpy array): Array of shape (n_out, n_h), provides hidden-to-
            output-layer weights.
        b_out (numpy array): Array of shape (n_out), provides the bias term
            in the output.
        param_names (list): The list of each parameter's name as a string,
            e.g. "W_rec" for a vanilla RNN or "W_a" in an LSTM.
        output (functions.Function): An instance of the Function class used
            for calculating final output from z.
        loss (functions.Function): An instance of the Function class used for
            calculating loss from z (must implicitly include output function,
            e.g. softmax_cross_entropy if output is softmax).
        x (numpy array): Array of shape (n_in) representing the current inputs
            to the network.
        state (numpy array): Array of shape (n_h) representing the network
            state, i.e. what is strictly necessary for running the network
            forwards (so not including e.g. pre-activations).
        full_state (dict): A dictionary of numpy arrays (names inherited from
            child class) that are needed for computing derivatives of the
            network.
        pre_output (numpy array): Array of shape (n_out) representing the
            outputs of the network, before any final output nonlinearities,
            e.g. softmax, are applied.
        error (numpy array): Array of shape (n_out) representing the derivative
            of the loss with respect to pre_output. Calculated by loss.f_prime
            in simulation.Simulation class when provided class labels.
        y_hat (numpy array): Array of shape (n_out) representing the final
            outputs of the network, to be directly compared with task labels.
        *_prev (numpy array): Array representing any of x, state, etc. at
            the previous time step."""

    def __init__(self, *args, **kwargs):
        """Initializes the RNN's trainable parameter values by specifying.
        Child class must provide list of parameter names."""

        ### --- Initialize weights from child class --- ###
        try:
            param_names = tuple(self.param_names) + ('W_out', 'b_out')
        except AttributeError:
            raise AssertionError('Child class must provide list of param names')

        initial_params = args[:len(param_names)]

        for param_name, initial_param in zip(param_names, initial_params):
            setattr(self, param_name, initial_param)

        ### --- Set output and loss functions --- ###
        try:
            self.output = args[-2]
            self.loss = args[-1]
        except IndexError:
            raise AssertionError('Must provide output and loss functions')

    def update_output(self, y_label):
        """Update outputs using current state of the network."""

        self.pre_output_prev = np.copy(self.pre_output)
        self.pre_output = self.W_out.dot(self.state) + self.b_out
        self.y_hat = self.output.f(self.pre_output)
        self.error = self.output.f_prime(self.pre_output, y_label)

    def get_flattened_params(self):
        """Get a single list of all parameter values for the network"""

        ret = [getattr(self, param).flatten() for param in self.param_names]
        ret = np.array(ret)
        return ret

    def update_network_history(self):
        """Update the history of the network with data needed to compute
        previous derivatives."""

        self.network_history.append(self.get_full_state())

class Vanilla_RNN(RNN):
    """A vanilla recurrent neural network.

    Obeys the forward equation (in TeX)

    h^t = W^{rec} a^{t-1} + W^{in}x^t + b^{rec}
    a^t = (1 - \alpha)a^{t-1} + \alpha \phi(h^t)

    Attributes:
        W_rec (numpy array): Array of shape (n_h, n_h_hat), weights recurrent
            inputs.
        W_in (numpy array): Array of shape (n_h, n_h_hat), weights recurrent
            inputs.
        b_rec (numpy array): Array of shape (n_h), represents the bias term
            in the recurrent update equation.
        activation (functions.Function): An instance of the Function class
            used as the network's nonlinearity \phi in the recurrent update
            equation.
        alpha (float): Ratio of time constant of integration to time constant
            of leak. Must be less than 1.
        loss (functions.Function): An instance of the Function class used for
            calculating loss from z (must implicitly include output function,
            e.g. softmax_cross_entropy if output is softmax).
        h (numpy array): Array of shape (n_h) representing the pre-activations
            of the network.
        a (numpy array): Array of shape (n_h) representing the post-activations
            of the network."""

    def __init__(self, W_rec, b_rec, W_out, b_out,
                 output, loss, activation, alpha):
        """Initializes an RNN by specifying its initial parameter values;
        its activation, output, and loss functions; and alpha."""

        self.param_names = ('W_rec', 'b_rec')

        super().__init__(W_rec, b_rec, W_out, b_out,
                         output, loss)

        #Network dimensions
        self.n_state = W_rec.shape[0]
        self.n_in = W_rec.shape[1] - W_rec.shape[0]
        self.n_state_hat = W_rec.shape[1]
        self.n_out = W_out.shape[0]
        self.m = self.n_state_hat + 1

        #Check dimension consistency.
        assert self.n_state == W_out.shape[1]
        assert self.n_state == b_rec.shape[0]
        assert self.n_out == b_out.shape[0]

        #Activation and loss functions
        self.alpha = alpha
        self.activation = activation

        #Number of parameters
        self.n_h_params = self.W_rec.size + self.b_rec.size

        #Regularization
        self.L1_reg = [True, False, True, False]
        self.L2_reg = [True, False, True, False]

        #Netowrk history
        self.network_history = []

        #Initial state values
        self.reset_network()

    def reset_network(self, sigma_reset=1, **kwargs):
        """Resets hidden state of the network, either randomly or by
        specifying with kwargs.

        Args:
            sigma (float): Standard deviation of (zero-mean) Gaussian random
                reset of pre-activations h. Used if neither h or state is
                specified.
            h (numpy array): The specification of the pre-activation values,
                must be of shape (self.n_state). The state values are determined
                by application of the nonlinearity to h.
            state (numpy array): The specification of the post-activation values,
                must be of shape (self.n_state). If not specified, determined
                by h."""

        if 'h' in kwargs.keys(): #Manual reset if specified.
            self.h = kwargs['h']
        else: #Random reset by sigma if not.
            self.h = np.random.normal(0, sigma_reset, self.n_h)

        self.state = self.activation.f(self.h) #Specify activations by \phi.

        if 'state' in kwargs.keys(): #Override with manual activations if given.
            self.state = kwargs['state']

        #Update outputs based on current state values
        self.update_output()

    def next_state(self, x, state=None, update=True, sigma=0):
        """Advances the network forward by one time step.

        Accepts as argument the current time step's input x and updates
        the state of the RNN, while storing the previous state h
        and activatation a. Can either update the network (if update=True)
        or return what the update would be.

        Args:
            x (numpy array): Input provided to the network, of shape (n_in).
            update (bool): Specifies whether to update the network using the
                current network state (if True) or return the would-be next
                network state using a provided "current" network state a.
            state (numpy array): Recurrent inputs used to drive the network,
                to be provided only if update is False.
            sigma (float): Standard deviation of white noise added to pre-
                activations before applying \phi.

        Returns:
            Updates self.x, self.h, self.state, and self.*_prev, or returns the
            would-be update from given previous state."""

        if state is None:
            state_prev = self.state.copy()
        else:
            state_prev = state
        state_hat = np.concatenate([state_prev, x])
        h = self.W_rec.dot(state_hat) + self.b_rec
        state = ((1 - self.alpha) * self.state +
                 self.alpha * self.activation.f(h))
        if sigma > 0: #Add noise if sigma is more than 0
            noise = np.random.normal(0, sigma, self.n_state)
            state += noise

        if update:
            self.x = x
            self.state_prev = state_prev
            self.h = h
            self.state_hat = state_hat
            self.state = state
        else:
            return state

    def get_state_jacobian(self, full_state=None):
        """Calculates the Jacobian of the network.

        Follows the equation
        J_{ij} = \alpha\phi'(h_i) W_{rec,ij} + (1 - \alpha)\delta_{ij}

        Args:
            full_state (dict): A dict specifying what the state of the network
                should be (in particular the pre-activations h) when calculating
                the Jacobian. If left as None, then the current state of the
                network is used.
        Returns:
            A numpy array of shape (n_state, n_state)."""

        if full_state is None:
            h = self.h
        else:
            h = full_state['h']

        #Calculate Jacobian
        D = self.activation.f_prime(h) #Nonlinearity derivative
        J_state = (self.alpha * (D * self.W_rec[:, :self.n_state].T).T +
                   (1 - self.alpha) * np.eye(self.n_state))

        return J_state

    def get_immediate_influence(self, full_state=None, sparse=False):
        """Calculate the immediate influence of the "recurrent" parameters
        on the network state."""

        if full_state is None:
            h = self.h
        else:
            h = full_state['h']
            state_hat = self.state_hat

        #Get relevant values and derivatives from network.
        state_hat_ = np.concatenate([state_hat, np.array([1])])

        if sparse:
            D = np.diag(self.activation.f_prime(h))
            papw = np.kron(state_hat_, D)
        else:
            D = self.activation.f_prime(h)
            papw = np.multiply.outer(D, state_hat_)

        return papw

    def get_full_state(self):
        """Return copies of the "full" network state, i.e. that which is
        needed to compute derivatives. Organized in a dictionary with keys
        corresponding to attribute names."""

        full_state = {'h': self.h.copy(),
                      'state_hat': self.state_hat.copy(),
                      'error': self.error.copy()}

        return full_state
