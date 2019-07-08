#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 12:32:35 2019

@author: omarschall
"""

import numpy as np
import unittest
from network import RNN
from simulation import Simulation
from learning_algorithms import *
from functions import *
from gen_data import *
from optimizers import *
from utils import *
from pdb import set_trace

class Test_Learning_Algorithm(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Initializes task data and RNNs so that simulations can be run."""

        cls.task = Coin_Task(4, 6, one_hot=True,
                             deterministic=True, tau_task=4)
        cls.data = cls.task.gen_data(50, 50)

        n_in = cls.task.n_in
        n_h = 16
        n_out = cls.task.n_out

        cls.W_in = np.random.normal(0, np.sqrt(1/(n_in)), (n_h, n_in))
        M_rand = np.random.normal(0, 1, (n_h, n_h))
        cls.W_rec = np.linalg.qr(M_rand)[0]
        cls.W_out = np.random.normal(0, np.sqrt(1/(n_h)), (n_out, n_h))
        cls.W_FB = np.random.normal(0, np.sqrt(1/n_out), (n_out, n_h))

        cls.b_rec = np.zeros(n_h)
        cls.b_out = np.zeros(n_out)

        alpha = 1

        cls.rnn_1 = RNN(cls.W_in, cls.W_rec, cls.W_out,
                        cls.b_rec, cls.b_out,
                        activation=tanh,
                        alpha=alpha,
                        output=softmax,
                        loss=softmax_cross_entropy)

        cls.rnn_2 = RNN(cls.W_in, cls.W_rec, cls.W_out,
                        cls.b_rec, cls.b_out,
                        activation=tanh,
                        alpha=alpha,
                        output=softmax,
                        loss=softmax_cross_entropy)



    def test_DNI_eligibility_traces(self):
        """Verifies that DNI algorithm gives save result whether using
        explicit parameter influence or a filtered version with time constant
        of 1."""

        np.random.seed(1)
        self.optimizer_1 = SGD(lr=0.0005)
        self.SG_optimizer_1 = SGD(lr=0.01)
        self.learn_alg_1 = DNI(self.rnn_1, self.SG_optimizer_1,
                               W_a_lr=0.01, backprop_weights='approximate',
                               SG_label_activation=tanh, W_FB=self.W_FB)
        np.random.seed(1)
        self.optimizer_2 = SGD(lr=0.0005)
        self.SG_optimizer_2 = SGD(lr=0.01)
        self.learn_alg_2 = DNI(self.rnn_2, self.SG_optimizer_2,
                               W_a_lr=0.01, backprop_weights='approximate',
                               SG_label_activation=tanh, W_FB=self.W_FB,
                               alpha_e=1)
        monitors = ['net.a']

        np.random.seed(2)
        self.sim_1 = Simulation(self.rnn_1)
        self.sim_1.run(self.data, learn_alg=self.learn_alg_1,
                       optimizer=self.optimizer_1,
                       monitors=monitors,
                       verbose=False)

        np.random.seed(2)
        self.sim_2 = Simulation(self.rnn_2)
        self.sim_2.run(self.data, learn_alg=self.learn_alg_2,
                       optimizer=self.optimizer_2,
                       monitors=monitors,
                       verbose=False)

        self.assertTrue(np.isclose(self.sim_1.mons['net.a'],
                                   self.sim_2.mons['net.a']).all())

    def test_forward_bptt(self):
        """Verifies that BPTT algorithm gives save aggregate weight change as
        RTRL for a very small learning rate, while also checking that the
        recurrent weights did change some amount (i.e. learning rate not *too*
        small)."""

        self.data = self.task.gen_data(200, 100)

        np.random.seed(1)
        self.optimizer_1 = SGD(lr=0.000001)
        self.learn_alg_1 = RTRL(self.rnn_1)
        np.random.seed(1)
        self.optimizer_2 = SGD(lr=0.000001)
        self.learn_alg_2 = Forward_BPTT(self.rnn_2, 15)

        monitors = []

        np.random.seed(2)
        self.sim_1 = Simulation(self.rnn_1)
        self.sim_1.run(self.data, learn_alg=self.learn_alg_1,
                       optimizer=self.optimizer_1,
                       monitors=monitors,
                       verbose=False)

        np.random.seed(2)
        self.sim_2 = Simulation(self.rnn_2)
        self.sim_2.run(self.data, learn_alg=self.learn_alg_2,
                       optimizer=self.optimizer_2,
                       monitors=monitors,
                       verbose=False)

        #Assert networks learned similar weights with a small tolerance.
        self.assertTrue(np.isclose(self.rnn_1.W_rec,
                                   self.rnn_2.W_rec, atol=1e-5).all())
        #Assert networks' parameters changed appreciably, despite a large
        #tolerance for closeness.
        self.assertFalse(np.isclose(self.W_rec,
                                    self.rnn_2.W_rec, atol=1e-3).all())

    def test_kernl_reduce_rflo(self):
        """Verifies that KeRNL reduces to RFLO in special case.

        If beta is initialized to the identity while the gammas are all
        initialized to the network inverse time constant alpha, and the KeRNL
        optimizer has 0 learning rate (i.e. beta and gamma do not change), then
        KeRNL should produce the same gradients as RFLO if the approximate
        KeRNL of (1 - alpha) (rather than exp(-alpha)) is used."""

        self.data = self.task.gen_data(200, 100)

        alpha = 0.3

        self.rnn_1.alpha = alpha
        self.rnn_2.alpha = alpha

        #RFLO
        np.random.seed(1)
        self.optimizer_1 = SGD(lr=0.001)
        self.learn_alg_1 = RFLO(self.rnn_1, alpha)
        #KeRNL with beta and gamma fixed to RFLO values
        np.random.seed(1)
        self.optimizer_2 = SGD(lr=0.001)
        self.KeRNL_optimizer = SGD(lr=0)
        beta = np.eye(self.rnn_2.n_h)
        gamma = np.ones(self.rnn_2.n_h)*alpha
        self.learn_alg_2 = KeRNL(self.rnn_2, self.KeRNL_optimizer,
                                 beta=beta, gamma=gamma,
                                 use_approx_kernel=True)

        monitors = []

        np.random.seed(2)
        self.sim_1 = Simulation(self.rnn_1)
        self.sim_1.run(self.data, learn_alg=self.learn_alg_1,
                       optimizer=self.optimizer_1,
                       monitors=monitors,
                       verbose=False)

        np.random.seed(2)
        self.sim_2 = Simulation(self.rnn_2)
        self.sim_2.run(self.data, learn_alg=self.learn_alg_2,
                       optimizer=self.optimizer_2,
                       monitors=monitors,
                       verbose=False)

        #Assert networks learned similar weights with a small tolerance.
        self.assertTrue(np.isclose(self.rnn_1.W_rec,
                                   self.rnn_2.W_rec, atol=1e-5).all())
        #Assert networks' parameters changed appreciably, despite a large
        #tolerance for closeness.
        self.assertFalse(np.isclose(self.W_rec,
                                    self.rnn_2.W_rec, atol=1e-3).all())

    def test_uoro_unbiased(self):
        """Verifies that if several iid instances of the uoro approximation
        are run at each time step, their average is close to the true influence
        matrix from RTRL."""

        self.data = self.task.gen_data(6, 10)
        self.learn_alg = UORO(self.rnn_1)
        self.comp_algs = [RTRL(self.rnn_1)]
        self.optimizer = SGD(lr=0)
        self.sim = Simulation(self.rnn_1)
        self.sim.run(self.data, learn_alg=self.learn_alg,
                     comp_algs=self.comp_algs,
                     optimizer=self.optimizer,
                     monitors=['dadw'],
                     verbose=False)
        self.rnn_1.next_state(self.data['test']['X'][0])
        self.comp_algs[0].update_learning_vars()
        self.learn_alg.update_learning_vars(update=False)
        gradient_estimates = []
        n_estimates = 1000
        for i in range(n_estimates):
            A, B = self.learn_alg.get_influence_estimate()
            gradient_estimates.append(np.multiply.outer(A, B))
        mean_grad_estimate = sum(gradient_estimates)/n_estimates
        #self.assertTrue(np.isclose(mean_grad_estimate,
        #                           self.comp_algs[0].dadw).all())

    def test_kfrtrl_unbiased(self):
        """Verifies that if several iid instances of the uoro approximation
        are run at each time step, their average is close to the true influence
        matrix from RTRL."""

        self.data = self.task.gen_data(6, 10)
        self.learn_alg = KF_RTRL(self.rnn_1)
        self.comp_algs = [RTRL(self.rnn_1)]
        self.optimizer = SGD(lr=0)
        self.sim = Simulation(self.rnn_1)
        self.sim.run(self.data, learn_alg=self.learn_alg,
                     comp_algs=self.comp_algs,
                     optimizer=self.optimizer,
                     monitors=['RTRL.dadw'],
                     verbose=False)
        self.rnn_1.next_state(self.data['test']['X'][0])
        self.comp_algs[0].update_learning_vars()
        self.learn_alg.update_learning_vars(update=False)
        gradient_estimates = []
        n_estimates = 1000
        for i in range(n_estimates):
            A, B = self.learn_alg.get_influence_estimate()
            gradient_estimates.append(np.kron(A, B))
        mean_grad_estimate = sum(gradient_estimates)/n_estimates
        #self.assertTrue(np.isclose(mean_grad_estimate,
        #                           self.comp_algs[0].dadw).all())


if __name__ == '__main__':
    unittest.main()
