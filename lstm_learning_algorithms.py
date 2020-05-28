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
from learning_algorithms import Learning_Algorithm, Stochastic_Algorithm,RTRL


class Only_Output_LSTM(RTRL):
    
    def __init__(self, rnn):
        
        super().__init__(rnn)
    
    def update_learning_vars(self):
        
        pass
    
    def get_rec_grads(self):
        
        dLdw = np.zeros((self.n_h, self.m))
        
        return dLdw

class Stochastic_Algorithm(Learning_Algorithm):

    def sample_nu(self):
        """Sample nu from specified distribution."""

        if self.nu_dist == 'discrete' or self.nu_dist is None:
            nu = np.random.choice([-1, 1], self.n_nu)
        elif self.nu_dist == 'gaussian':
            nu = np.random.normal(0, 1, self.n_nu)
        elif self.nu_dist == 'uniform':
            nu = np.random.uniform(-1, 1, self.n_nu)

        return nu


class UORO_LSTM(Stochastic_Algorithm):
    """Implements the Unbiased Online Recurrent Optimization (UORO) algorithm
    from Tallec et al. 2017.

    Full details in our review paper or in original paper. Broadly, an outer
    product approximation of M is maintained by 2 vectors A and B, which update
    by the equations

    A' = p0 J A + p1 \nu        (1)
    B' = 1/p0 B + 1/p1 \nu M_immediate      (2)

    where \nu is a vector of zero-mean iid samples. p0 and p1 are calculated by

    p0 = \sqrt{norm(B)/norm(A)}       (3)
    p1 = \sqrt{norm(\nu papw)/norm(\nu)}        (4)

    These equations are implemented in update_learning_vars by two different
    approaches. If 'epsilon' is provided as an argument, then the "forward
    differentiation" method from the original paper is used, where the matrix-
    vector product JA is estimated numerically by a perturbation of size
    epsilon in the A direction.

    Then the recurrent gradients are calculated by

    dL/dw = qM = (q A) B    (5)

    Eq. (5) is implemented in the get_rec_grads method."""

    def __init__(self, rnn, **kwargs):
        """Inits an UORO instance by setting the initial values of A and B to be
        iid samples from a standard normal distribution, to avoid dividing by
        zero in Eqs. (3) and (4).

        Keyword args:
            epsilon (float): Scaling factor on perturbation for forward
                differentiation method. If not provided, exact derivative is
                calculated instead.
            P0 (float): Overrides calculation of p0, instead uses provided value
                of P0. If not provided, p0 is calculated according to Eq. (3).
            P1 (float): Same for p1.
            A (numpy array): Initial value for A.
            B (numpy array): Initial value for B.
            nu_dist (string): Takes on the value of 'gaussian', 'discrete', or
                'uniform' to indicate what type of distribution nu should sample
                 from. Default is 'discrete'."""

        self.name = 'UORO' #Default algorithm name
        allowed_kwargs_ = {'epsilon', 'P0', 'P1', 'A_f', 'B_f','A_i', 'B_i',
        'A_a', 'B_a','A_o', 'B_o', 'nu_dist'}
        super().__init__(rnn, allowed_kwargs_, **kwargs)
        self.n_nu = self.n_t 

        #Initialize A and B arrays
        self.A_dict = {'f':self.A_f,'i':self.A_i,'a':self.A_a,'o':self.A_o}
        self.B_dict = {'f':self.B_f, 'i':self.B_i,'a':self.B_a,'o':self.B_o}

        for a in self.A_dict:
            if self.A_dict[a] is None:
                self.A_dict[a] = np.random.normal(0, 1, self.n_t) 

        for b in self.B_dict:
            if self.B_dict[b] is None:
                self.B_dict[b] = np.random.normal(0, 1, (self.n_h, self.n_h+self.n_in+1)) 

    def update_learning_vars(self, update=True):
        """Implements Eqs. (1), (2), (3), and (4) to update the outer product
        approximation of the influence matrix by A and B.

        Args:
            update (bool): If True, updates the algorithm's current outer
                product approximation B, A. If False, only prepares for calling
                get_influence_estimate."""

        self.rnn.get_a_jacobian() #Get updated network Jacobian
        self.rnn.update_compact_M_immediate() #Get immediate influence papw
        print('self.rnn.papwf: ',self.rnn.papwf_c.shape)
        A_f, B_f = self.get_influence_estimate(self.rnn.papwf_c,'f')
        A_i, B_i = self.get_influence_estimate(self.rnn.papwi_c,'i')
        A_a, B_a = self.get_influence_estimate(self.rnn.papwa_c,'a')
        A_o, B_o = self.get_influence_estimate(self.rnn.papwo_c,'o')

        if update:
            self.A_f, self.B_f = A_f, B_f
            self.A_i, self.B_i = A_i, B_i
            self.A_a, self.B_a = A_a, B_a
            self.A_o, self.B_o = A_o, B_o

    def get_influence_estimate(self,papw,gate):
        """Generates one random outer-product estimate of the influence matrix.

        Samples a random vector nu of iid samples with 0 mean from a
        distribution given by nu_dist, and returns an updated estimate
        of A and B from Eqs. (1)-(4).

        Returns:
            Updated A (numpy array of shape (n_h)) and B (numpy array of shape
                (n_h, m))."""

        #Sample random vector
        self.nu = self.sample_nu()

        #Get random projection of M_immediate onto \nu
        M_projection = (papw.T*self.nu).T 
        print('M_projection: ',M_projection.shape)

        if self.epsilon is not None: #Forward differentiation method
            eps = self.epsilon
            #Get perturbed state in direction of A
            self.c_perturbed = self.rnn.c_prev + eps * self.A_dict[gate][:self.n_t]
            self.h_perturbed = self.rnn.h_prev + eps * self.A_dict[gate][self.n_t:]
            #Get hypothetical next states from this perturbation
            self.a_perturbed_next = self.rnn.next_state(self.rnn.x,
                                                        self.c_perturbed,
                                                        self.h_preturbed,
                                                        update=False)
            #Get forward-propagated A
            c_h = np.concatenate([self.rnn.c, self.rnn.h])
            self.A_forwards = (self.a_perturbed_next - c_h) / eps
            #Calculate scaling factors
            B_norm = norm(self.B_dict[gate])
            A_norm = norm(self.A_forwards)
            M_norm = norm(M_projection)
            self.p0 = np.sqrt(B_norm/(A_norm + eps)) + eps
            self.p1 = np.sqrt(M_norm/(np.sqrt(self.n_t) + eps)) + eps
        else: #Backpropagation method
            #Get forward-propagated A
            self.A_forwards = self.rnn.a_J.dot(self.A_dict[gate])
            #Calculate scaling factors
            B_norm = norm(self.B_dict[gate])
            A_norm = norm(self.A_forwards)
            M_norm = norm(M_projection)
            self.p0 = np.sqrt(B_norm/A_norm)
            self.p1 = np.sqrt(M_norm/np.sqrt(self.n_t))

        #Override with fixed P0 and P1 if given
        if self.P0 is not None:
            self.p0 = np.copy(self.P0)
        if self.P1 is not None:
            self.p1 = np.copy(self.P1)

        #Update outer product approximation
        A = self.p0 * self.A_forwards + self.p1 * self.nu
        B = (1/self.p0) * self.B_dict[gate] + (1 / self.p1) * M_projection

        return A, B

    def get_rec_grads(self):
        """Calculates recurrent grads by taking matrix product of q with the
        estimate of the influence matrix.

        First associates q with A to calculate a "global learning signal"
        Q, which multiplies by B to compute the recurrent gradient, which
        is reshaped into original matrix form.

        Returns:
            An array of shape (n_h, m) representing the recurrent gradient."""

        dLdw_f = self.q.dot(self.A_f) * self.B_f
        dLdw_i = self.q.dot(self.A_i) * self.B_i
        dLdw_a = self.q.dot(self.A_a) * self.B_a
        dLdw_o = self.q.dot(self.A_o) * self.B_o
        dLdw = np.concatenate([dLdw_f, dLdw_i, dLdw_a, dLdw_o],axis=1)
        print('self.B_f',self.B_f.shape)
        print('dLdw',dLdw.shape)
        print('q',self.q.shape)
        return dLdw

    def reset_learning(self):
        """Resets learning by re-randomizing the outer product approximation to
        random gaussian samples."""

        for a in self.A_dict:
            self.A_dict[a]= np.random.normal(0, 1, self.n_t)
        for b in self.B_dict:
            self.B_dict[b] = np.random.normal(0, 1, (self.n_h, self.n_h+self.n_in+1))

    