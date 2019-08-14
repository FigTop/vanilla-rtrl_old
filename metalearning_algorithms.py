#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  13 13:17:11 2019

@author: omarschall
"""

import numpy as np
from pdb import set_trace
from utils import *
from functions import *
from copy import copy
from learning_algorithms import Real_Time_Learning_Algorithm

class Forward_BPTT_LR_by_RTRL(Real_Time_Learning_Algorithm):

    def __init__(self, net, optimizer, T_truncation,
                 meta_lr=0.01, H_epsilon=0.001, **kwargs):

        self.name = 'F-BPTT'
        allowed_kwargs_ = set()
        super().__init__(net, allowed_kwargs_, **kwargs)

        self.T_truncation = T_truncation
        self.meta_lr = meta_lr
        self.optimizer = optimizer
        self.H_epsilon = H_epsilon
        self.CA_hist = []
        self.a_hat_hist = []
        self.h_hist = []
        self.q_hist = []
        self.Gamma = np.zeros((self.n_h, self.m))

    def update_learning_vars(self):

        #Initialize new credit assignment for current time step
        self.CA_hist.append([0])

        #Update history
        self.a_hat = np.concatenate([self.net.a_prev, self.net.x, np.array([1])])
        self.a_hat_hist.append(np.copy(self.a_hat))
        self.h_hist.append(np.copy(self.net.h))
        self.q_hist.append(np.copy(self.q))

        self.propagate_feedback_to_hidden()

        q = np.copy(self.q)

        for i_BP in range(len(self.CA_hist)):

            self.CA_hist[-(i_BP + 1)] += q
            J = self.net.get_a_jacobian(update=False,
                                        h=self.h_hist[-(i_BP+1)])
            q = q.dot(J)

    def get_approximate_hessian_vector_product(self):

        Gamma_rec = self.Gamma[:, :self.n_h]
        Gamma_in = self.Gamma[:, self.n_h:self.n_h + self.n_in]
        Gamma_b = self.Gamma[:, -1]
        perturbed_net_1 = copy(self.net)
        perturbed_net_2 = copy(self.net)
        perturbed_net_1.W_rec += self.H_epsilon * Gamma_rec
        perturbed_net_1.W_in += self.H_epsilon * Gamma_in
        perturbed_net_1.b_rec += self.H_epsilon * Gamma_b
        perturbed_net_2.W_rec -= self.H_epsilon * Gamma_rec
        perturbed_net_2.W_in -= self.H_epsilon * Gamma_in
        perturbed_net_2.b_rec -= self.H_epsilon * Gamma_b


        perturbed_CA_1 = 0
        perturbed_CA_2 = 0

        for i_q in range(len(self.q_hist) - 1, -1, -1):

            q_1 = np.copy(self.q_hist[i_q])
            q_2 = np.copy(q_1)

            for i_BP in range(i_q, -1, -1):

                perturbed_CA_1 += q_1
                perturbed_CA_2 += q_2
                J_1 = perturbed_net_1.get_a_jacobian(update=False,
                                                     h=self.h_hist[i_BP])
                J_2 = perturbed_net_2.get_a_jacobian(update=False,
                                                     h=self.h_hist[i_BP])
                q_1 = q_1.dot(J_1)
                q_2 = q_2.dot(J_2)

            perturbed_CA_1 += q_1
            perturbed_CA_2 += q_2

        D = self.net.activation.f_prime(self.h_hist[0])
        grad_1 = np.multiply.outer(perturbed_CA_1 * D, self.a_hat_hist[0])
        grad_2 = np.multiply.outer(perturbed_CA_2 * D, self.a_hat_hist[0])
        hessian_vector_product = (grad_1 - grad_2)/(2 * self.H_epsilon)

        return hessian_vector_product

    def get_rec_grads(self):

        self.D = self.net.activation.f_prime(self.h_hist[0])

        if len(self.CA_hist)==self.T_truncation:

            self.net.CA = np.copy(self.CA_hist[0])

            #self.D = self.net.activation.f_prime(self.h_hist[0])
            rec_grads = np.multiply.outer(self.net.CA * self.D, self.a_hat_hist[0])

            self.delete_history()

        else:

            rec_grads = np.zeros((self.n_h, self.m))

        # UPDATE GAMMA AND OPTIMIZER LEARNING RATE
        HVP = self.get_approximate_hessian_vector_product()
        self.Gamma = self.Gamma - HVP - rec_grads

        self.eta_error = np.multiply.outer(self.q_hist[0] * self.D,
                                           self.a_hat_hist[0])
        self.eta_grad = (self.eta_error * self.Gamma).sum()
        self.optimizer.lr -= self.meta_lr * self.eta_grad

        return rec_grads

    def reset_learning(self):

        self.CA_hist = []
        self.a_hat_hist = []
        self.h_hist = []
        self.q_hist = []

    def delete_history(self):

        for attr in ['CA', 'a_hat', 'h', 'q']:
            del(self.__dict__[attr+'_hist'][0])