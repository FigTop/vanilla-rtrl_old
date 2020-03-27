#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 15:03:11 2018

@author: omarschall
"""

import numpy as np
from pdb import set_trace
from utils import *
from functions import *
from copy import deepcopy
from learning_algorithms import Learning_Algorithm

class RTRL_LSTM(Learning_Algorithm):
    
    def __init__(self, lstm, **kwargs):
        """Inits an RTRL instance by setting the initial dadw matrix to zero."""

        self.name = 'RTRL_LSTM' #Algorithm name
        allowed_kwargs_ = set()
        allowed_kwargs = {'W_FB', 'L2_reg'}.union(allowed_kwargs_)

        for k in kwargs:
            if k not in allowed_kwargs:
                raise TypeError('Unexpected keyword argument passed'
                                'to Learning_Algorithm.__init__: ' + str(k))

        #Set all non-specified kwargs to None
        for attr in allowed_kwargs:
            if not hasattr(self, attr):
                setattr(self, attr, None)

        #Make kwargs attributes of the instance
        self.__dict__.update(kwargs)

        #Define basic learning algorithm properties
        self.lstm = lstm
        self.n_in = self.lstm.n_in
        self.n_h = self.lstm.n_h
        self.n_t = self.lstm.n_t
        self.n_out = self.lstm.n_out
        self.m = self.n_h + self.n_in + 1
        self.q = np.zeros(self.n_t)

        #Initialize influence matrix
        self.dadwf = np.zeros((self.n_t, self.lstm.n_h_params))
        self.dadwi = np.zeros((self.n_t, self.lstm.n_h_params))
        self.dadwa = np.zeros((self.n_t, self.lstm.n_h_params))
        self.dadwo = np.zeros((self.n_t, self.lstm.n_h_params))

    def update_learning_vars(self):
        """Updates the influence matrix via Eq. (1)."""


        self.a_hat = np.concatenate([self.lstm.h_hat_prev, np.array([1])])
        
        #Calculate M_immediate
        
        r = self.lstm.o * self.lstm.tanh.f_prime(self.lstm.c)

        # self.m = n_h_prev+1
        pcpwo = np.zeros((self.n_h, self.m * self.n_h))
        D_o = np.diag(self.lstm.sigmoid.f_prime(self.lstm.o) * self.lstm.tanh.f(self.lstm.c))
        phpwo = np.kron(self.a_hat, D_o)
        self.papwo = np.concatenate([pcpwo, phpwo])

        D_a = np.diag(self.lstm.tanh.f_prime(self.lstm.a) * self.lstm.i)
        pcpwa = np.kron(self.a_hat, D_a)
        phpwa = (pcpwa.T * r).T
        self.papwa = np.concatenate([pcpwa, phpwa])

        D_i = np.diag(self.lstm.sigmoid.f_prime(self.lstm.i) * self.lstm.a)
        pcpwi = np.kron(self.a_hat, D_i)
        phpwi = (pcpwi.T * r).T
        self.papwi = np.concatenate([pcpwi, phpwi])

        D_f = np.diag(self.lstm.sigmoid.f_prime(self.lstm.f) * self.lstm.c_prev)
        pcpwf = np.kron(self.a_hat, D_f)
        phpwf = (pcpwf.T * r).T
        self.papwf = np.concatenate([pcpwf, phpwf])

        self.lstm.get_a_jacobian() #Get updated network Jacobian

        #Update influence matrix via Eq. (1).
        """dimension of dadwf (n_t, m * n_h)"""
        self.dadwf = self.lstm.a_J.dot(self.dadwf) + self.papwf
        self.dadwi = self.lstm.a_J.dot(self.dadwi) + self.papwi
        self.dadwa = self.lstm.a_J.dot(self.dadwa) + self.papwa
        self.dadwo = self.lstm.a_J.dot(self.dadwo) + self.papwo

    def get_rec_grads(self):
        """Calculates recurrent grads using Eq. (2), reshapes into original
        matrix form."""

        """ dL/dw = dL/da * da/dw
            dimension of q : n_t
            dimension of da/dw : (n_t, m * n_h)
        """

        dLdw_f = self.q.dot(self.dadwf).reshape((self.n_h, self.m), order='F')
        dLdw_o = self.q.dot(self.dadwo).reshape((self.n_h, self.m), order='F')
        dLdw_a = self.q.dot(self.dadwa).reshape((self.n_h, self.m), order='F')
        dLdw_i = self.q.dot(self.dadwi).reshape((self.n_h, self.m), order='F')
        
        return dLdw_f, dLdw_i, dLdw_a, dLdw_o

    def get_outer_grads(self):
        """Calculates the derivative of the loss with respect to the output
        parameters lstm.W_c_out, lstm.W_h_out and lstm.b_out.

        Calculates the outer gradients in the manner of a perceptron derivative
        by taking the outer product of the error with the "regressors" onto the
        output (the hidden state and constant 1).

        Returns:
            A numpy array of shape (rnn.n_out, self.n_h + 1) containing the
                concatenation (along column axis) of the derivative of the loss
                w.r.t. rnn.W_out and w.r.t. rnn.b_out."""

        self.state_ = np.concatenate([self.lstm.c,
                                      self.lstm.h,
                                      np.array([1])])
        return np.multiply.outer(self.lstm.error, self.state_)
    
    def propagate_feedback_to_hidden(self):
        """Performs one step of backpropagation from the outer-layer errors to
        the hidden state.

        Calculates the immediate derivative of the loss with respect to the
        hidden state rnn.a. By default, this is done by taking rnn.error (dL/dz)
        and applying the chain rule, i.e. taking its matrix product with the
        derivative dz/da, which is rnn.W_out. Alternatively, if 'W_FB' attr is
        provided to the instance, then these feedback weights, rather the W_out,
        are used, as in feedback alignment. (See Lillicrap et al. 2016.)

        Updates q to the current value of dL/da."""

        self.q_prev = np.copy(self.q)

        if self.W_FB is None:
            W_out = np.concatenate([self.lstm.W_c_out,
                                    self.lstm.W_h_out], axis=1)
            self.q = self.lstm.error.dot(W_out)
        else:
            self.q = self.lstm.error.dot(self.W_FB)

    def L2_regularization(self, grads):
        """Adds L2 regularization to the gradient.

        Args:
            grads (list): List of numpy arrays representing gradients before L2
                regularization is applied.
        Returns:
            A new list of grads with L2 regularization applied."""

        #Get parameters affected by L2 regularization
        L2_params = [self.lstm.params[i] for i in self.lstm.L2_indices]
        #Add to each grad the corresponding weight's current value, weighted
        #by the L2_reg hyperparameter.
        for i_L2, W in zip(self.lstm.L2_indices, L2_params):
            grads[i_L2] += self.L2_reg * W
        #Calculate L2 loss for monitoring purposes
        self.L2_loss = 0.5*sum([norm(p) for p in L2_params])
        return grads
    
    def __call__(self):
        """Calculates the final list of grads for this time step.

        Assumes the user has already called self.update_learning_vars, a
        method specific to each child class of Real_Time_Learning_Algorithm
        that updates internal learning variables, e.g. the influence matrix of
        RTRL. Then calculates the outer grads (gradients of W_out and b_out),
        updates q using propagate_feedback_to_hidden, and finally calling the
        get_rec_grads method (specific to each child class) to get the gradients
        of W_rec, W_in, and b_rec as one numpy array with shape (n_h, m). Then
        these gradients are split along the column axis into a list of 5
        gradients for W_rec, W_in, b_rec, W_out, b_out. L2 regularization is
        applied if L2_reg parameter is not None.

        Returns:
            List of gradients for W_rec, W_in, b_rec, W_out, b_out."""

        self.outer_grads = self.get_outer_grads()
        self.propagate_feedback_to_hidden()
        self.rec_grads = self.get_rec_grads()
        rec_grads_list = [split_weight_matrix(rg,[self.n_h + self.n_in, 1]) for rg in self.rec_grads]
        outer_grads_list = split_weight_matrix(self.outer_grads,
                                               [self.n_h, self.n_h, 1])
        grads_list = [g for G in rec_grads_list for g in G] + outer_grads_list

        if self.L2_reg is not None:
            grads_list = self.L2_regularization(grads_list)

        return grads_list

    def reset_learning(self):
        """Resets learning algorithm by setting influence matrix to 0."""

        self.dadwf *= 0
        self.dadwo *= 0
        self.dadwa *= 0
        self.dadwi *= 0

class Only_Output_LSTM(RTRL_LSTM):
    
    def __init__(self, lstm):
        
        super().__init__(lstm)
    
    def update_learning_vars(self):
        
        pass
    
    def get_rec_grads(self):
        
        dLdw_f = np.zeros((self.n_h, self.m))
        dLdw_o = np.zeros((self.n_h, self.m))
        dLdw_a = np.zeros((self.n_h, self.m))
        dLdw_i = np.zeros((self.n_h, self.m))
        
        return dLdw_f, dLdw_i, dLdw_a, dLdw_o