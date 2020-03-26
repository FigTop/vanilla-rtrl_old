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

class RTRL_LSTM(Learning_Algorithm):
    def __init__(self, lstm, **kwargs):
        """Inits an RTRL instance by setting the initial dadw matrix to zero."""

        self.name = 'RTRL_LSTM' #Algorithm name
        allowed_kwargs_ = set() #No special kwargs for RTRL
        super().__init__(lstm, allowed_kwargs_, **kwargs)

        #Initialize influence matrix
        self.dadwf = np.zeros((self.n_t, self.lstm.n_h_params))
        #self.dhdwf =
        self.dadwi = np.zeros((self.n_t, self.lstm.n_h_params))
        #self.dhdwo
        self.dadwa = np.zeros((self.n_t, self.lstm.n_h_params))
        #self.dhdwa
        self.dadwo = np.zeros((self.n_t, self.lstm.n_h_params))
        #self.dhdwi

    def update_learning_vars(self):
        """Updates the influence matrix via Eq. (1)."""


        self.a_hat = np.concatenate([self.lstm.h_hat_prev,
                                     np.array([1])])
        #Calculate M_immediate

        r = np.tile(self.lstm.o * self.lstm.tanh.f_prime(self.lstm.c),(self.n_h,1))

        # self.m = n_h_prev+1
        pcpwo = np.zeros((self.n_h,self.m * self.n_h))
        D_o = np.diag(self.lstm.sigmoid.f_prime(self.lstm.o) * self.lstm.tanh(self.lstm.c))
        phpwo = np.kron(self.a_hat,D_o)
        self.papwo = np.concatenate([pcpwo,phpwo])

        D_a = np.diag(self.lstm.tanh.f_prime(self.lstm.a) * self.lstm.i)
        pcpwa = np.kron(self.a_hat,D_a)
        phpwa = (pcpwa.T * r).T
        self.papwa = np.concatenate([pcpwa,phpwa])

        D_i = np.diag(self.lstm.sigmoid.f_prime(self.lstm.i)) * self.lstm.a)
        pcpwi = np.kron(self.a_hat,D_i)
        phpwi = (pcpwi.T * r).T
        self.papwi = np.concatenate([pcpwi,phpwi])

        D_f = np.diag(self.lstm.sigmoid.f_prime(self.lstm.f) * self.lstm.prev_c)
        pcpwf = np.kron(self.a_hat,D_f)
        phpwf = (pcpwf.T * r).T
        self.papwf = np.concatenate([pcpwf,phpwf])

        self.rnn.get_a_jacobian() #Get updated network Jacobian

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
        return dLdw_f,dLdw_o,dLdw_a,dLdw_i

    def reset_learning(self):
        """Resets learning algorithm by setting influence matrix to 0."""

        self.dadwf *= 0
        self.dadwo *= 0
        self.dadwa *= 0
        self.dadwi *= 0
