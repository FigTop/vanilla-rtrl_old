#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 12:32:35 2019

@author: omarschall
"""

import numpy as np
from numpy.testing import assert_allclose
import unittest
from unittest.mock import MagicMock
from network import RNN
from simulation import Simulation
from learning_algorithms import *
from functions import *
from gen_data import *
from optimizers import *
from pdb import set_trace

class Test_Learning_Algorithm(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        cls.W_in = np.eye(2)
        cls.W_rec = np.eye(2)
        cls.W_out = np.eye(2)
        cls.W_FB = -np.ones((2, 2)) + np.eye(2)
        cls.b_rec = np.zeros(2)
        cls.b_out = np.zeros(2)

        cls.rnn = RNN(cls.W_in, cls.W_rec, cls.W_out,
                      cls.b_rec, cls.b_out,
                      activation=identity,
                      alpha=1,
                      output=softmax,
                      loss=softmax_cross_entropy)

        cls.rnn.a = np.ones(2)
        cls.rnn.error = np.ones(2) * 0.5

    def test_get_outer_grads(self):

        self.learn_alg = Learning_Algorithm(self.rnn)
        outer_grads = self.learn_alg.get_outer_grads()
        correct_outer_grads = np.ones((2, 3)) * 0.5
        assert_allclose(outer_grads, correct_outer_grads)

    def test_propagate_feedback_to_hidden(self):

        #Case with symmetric feedback
        self.learn_alg = Learning_Algorithm(self.rnn)
        self.learn_alg.propagate_feedback_to_hidden()
        correct_q = np.ones(2) * 0.5
        assert_allclose(self.learn_alg.q, correct_q)

        #Case with random feedback
        self.learn_alg = Learning_Algorithm(self.rnn, W_FB=self.W_FB)
        self.learn_alg.propagate_feedback_to_hidden()
        correct_q = -np.ones(2) * 0.5
        assert_allclose(self.learn_alg.q, correct_q)

    def test_L2_regularization(self):

        self.learn_alg = Learning_Algorithm(self.rnn, L2_reg=0.1)
        grads = [np.zeros_like(p) for p in self.rnn.params]
        reg_grads = self.learn_alg.L2_regularization(grads)
        correct_reg_grads = [np.eye(2) * 0.1, np.eye(2) * 0.1,
                                     np.zeros(2), np.eye(2) * 0.1, np.zeros(2)]
        for grad, correct_grad in zip(reg_grads, correct_reg_grads):
            assert_allclose(grad, correct_grad)

    def test_call(self):

        self.learn_alg = Learning_Algorithm(self.rnn)
        self.learn_alg.get_rec_grads = MagicMock()
        self.learn_alg.get_rec_grads.return_value = np.ones((2, 5))
        grads = self.learn_alg()

        correct_grads = [np.ones((2, 2)), np.ones((2, 2)), np.ones(2),
                         np.ones((2, 2)) * 0.5, np.ones(2) * 0.5]

        for grad, correct_grad in zip(grads, correct_grads):
            assert_allclose(grad, correct_grad)

class Test_RTRL(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        cls.W_in = np.eye(2)
        cls.W_rec = np.eye(2)
        cls.W_out = np.eye(2)
        cls.b_rec = np.zeros(2)
        cls.b_out = np.zeros(2)

        cls.rnn = RNN(cls.W_in, cls.W_rec, cls.W_out,
                      cls.b_rec, cls.b_out,
                      activation=identity,
                      alpha=1,
                      output=softmax,
                      loss=softmax_cross_entropy)

        cls.rnn.a = np.ones(2)
        cls.rnn.a_prev = np.ones(2)
        cls.rnn.x = np.ones(2) * 2
        cls.rnn.error = np.ones(2) * 0.5
        cls.rnn.a_J = np.eye(2)

    def test_update_learning_vars(self):

        self.learn_alg = RTRL(self.rnn)
        self.learn_alg.dadw += 1
        self.learn_alg.update_learning_vars()
        papw = np.concatenate([np.eye(2), np.eye(2),
                               np.eye(2) * 2, np.eye(2) * 2, np.eye(2)], axis=1)
        correct_dadw = np.ones((2, 10)) + papw
        assert_allclose(self.learn_alg.dadw, correct_dadw)

    def test_get_rec_grads(self):

        self.learn_alg = RTRL(self.rnn)
        self.learn_alg.q = np.ones(2)
        self.learn_alg.dadw = np.concatenate([np.eye(2) * i for i in range(5)],
                                             axis=1)
        rec_grads = self.learn_alg.get_rec_grads()
        correct_rec_grads = np.array([list(range(5)) for _ in [0, 1]])
        assert_allclose(rec_grads, correct_rec_grads)

class Test_UORO(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        cls.W_in = np.eye(2)
        cls.W_rec = np.eye(2)
        cls.W_out = np.eye(2)
        cls.b_rec = np.zeros(2)
        cls.b_out = np.zeros(2)

        cls.rnn = RNN(cls.W_in, cls.W_rec, cls.W_out,
                      cls.b_rec, cls.b_out,
                      activation=identity,
                      alpha=1,
                      output=softmax,
                      loss=softmax_cross_entropy)

        cls.rnn.a = np.ones(2)
        cls.rnn.a_prev = np.ones(2)
        cls.rnn.x = np.ones(2) * 2
        cls.rnn.error = np.ones(2) * 0.5

    def test_update_learning_vars(self):

        self.learn_alg = UORO(self.rnn)
        self.learn_alg.get_influence_estimate = MagicMock()
        A, B = np.ones(2), np.ones((2, 5))
        self.learn_alg.get_influence_estimate.return_value = [A, B]
        self.learn_alg.update_learning_vars()

        correct_a_J = np.eye(2)
        correct_papw = np.array([[1, 1, 2, 2, 1],
                                 [1, 1, 2, 2, 1]])

        assert_allclose(self.rnn.a_J, correct_a_J)
        assert_allclose(self.learn_alg.papw, correct_papw)

    def test_get_influence_estimate(self):

        #Backpropagation method case
        A, B = np.ones(2), np.ones((2, 5))
        self.learn_alg = UORO(self.rnn, A=A, B=B)
        self.learn_alg.a_hat = np.array([1, 1, 2, 2, 1])
        self.learn_alg.papw = np.array([[1, 1, 2, 2, 1],
                                        [1, 1, 2, 2, 1]])
        self.rnn.a_J = np.eye(2)
        self.learn_alg.sample_nu = MagicMock()
        self.learn_alg.sample_nu.return_value = np.array([1, -1])
        A, B = self.learn_alg.get_influence_estimate()

        p0, p1 = np.sqrt(np.sqrt(5)), np.sqrt(np.sqrt(11))
        M_proj = np.array([[1, 1, 2, 2, 1],
                           [-1, -1, -2, -2, -1]])
        A_correct = p0 * np.array([1, 1]) + p1 * np.array([1, -1])
        B_correct = (1 / p0) * np.ones((2, 5)) + (1 / p1) * M_proj

        assert_allclose(A, A_correct)
        assert_allclose(B, B_correct)

    def test_get_rec_grads(self):

        A, B = np.ones(2), np.ones((2, 5))
        self.learn_alg = UORO(self.rnn, A=A, B=B)
        self.learn_alg.q = np.ones(2) * 0.5
        rec_grads = self.learn_alg.get_rec_grads()

        correct_rec_grads = np.ones((2, 5))

        assert_allclose(rec_grads, correct_rec_grads)

class Test_KF_RTRL(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        cls.W_in = np.eye(2)
        cls.W_rec = np.eye(2)
        cls.W_out = np.eye(2)
        cls.b_rec = np.zeros(2)
        cls.b_out = np.zeros(2)

        cls.rnn = RNN(cls.W_in, cls.W_rec, cls.W_out,
                      cls.b_rec, cls.b_out,
                      activation=identity,
                      alpha=1,
                      output=softmax,
                      loss=softmax_cross_entropy)

        cls.rnn.a = np.ones(2)
        cls.rnn.a_prev = np.ones(2)
        cls.rnn.x = np.ones(2) * 2
        cls.rnn.error = np.ones(2) * 0.5

    def test_update_learning_vars(self):

        A, B = np.ones(5), np.ones((2, 2))
        self.learn_alg = KF_RTRL(self.rnn, A=A, B=B)
        self.learn_alg.get_influence_estimate = MagicMock()
        self.learn_alg.get_influence_estimate.return_value = [A, B]
        self.learn_alg.update_learning_vars()

        correct_B_forwards = np.ones((2, 2))
        correct_D = np.eye(2)

        assert_allclose(self.learn_alg.B_forwards, correct_B_forwards)
        assert_allclose(self.learn_alg.D, correct_D)

    def test_get_influence_estimate(self):

        A, B = np.ones(5), np.ones((2, 2))
        self.learn_alg = KF_RTRL(self.rnn, A=A, B=B)
        self.learn_alg.a_hat = np.array([1, 1, 2, 2, 1])
        self.rnn.a_J = np.eye(2)
        self.learn_alg.D = np.eye(2)
        self.learn_alg.B_forwards = np.ones((2, 2))
        self.learn_alg.sample_nu = MagicMock()
        self.learn_alg.sample_nu.return_value = np.array([1, -1])
        A, B = self.learn_alg.get_influence_estimate()

        p0, p1 = np.sqrt(2/np.sqrt(5)), np.sqrt(np.sqrt(2/11))
        A_correct = p0 * np.ones(5) - p1 * np.array([1, 1, 2, 2, 1])
        B_correct = (1 / p0) * np.ones((2, 2)) - (1 / p1) * np.eye(2)
        assert_allclose(A, A_correct)
        assert_allclose(B, B_correct)

    def test_get_rec_grads(self):

        A, B = list(range(5)), np.ones((2, 2))
        self.learn_alg = KF_RTRL(self.rnn, A=A, B=B)
        self.learn_alg.q = np.ones(2) * 0.5
        rec_grads = self.learn_alg.get_rec_grads()

        correct_rec_grads = np.array([list(range(5)) for _ in [0, 1]])

        assert_allclose(rec_grads, correct_rec_grads)

class Test_Reverse_KF_RTRL(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        cls.W_in = np.eye(2)
        cls.W_rec = np.eye(2)
        cls.W_out = np.eye(2)
        cls.b_rec = np.zeros(2)
        cls.b_out = np.zeros(2)

        cls.rnn = RNN(cls.W_in, cls.W_rec, cls.W_out,
                      cls.b_rec, cls.b_out,
                      activation=identity,
                      alpha=1,
                      output=softmax,
                      loss=softmax_cross_entropy)

        cls.rnn.a = np.ones(2)
        cls.rnn.a_prev = np.ones(2)
        cls.rnn.x = np.ones(2) * 2
        cls.rnn.error = np.ones(2) * 0.5

    def test_update_learning_vars(self):

        A, B = np.ones(2), np.ones((2, 5))
        self.learn_alg = Reverse_KF_RTRL(self.rnn, A=A, B=B)
        self.learn_alg.get_influence_estimate = MagicMock()
        self.learn_alg.get_influence_estimate.return_value = [A, B]
        self.learn_alg.update_learning_vars()

        correct_B_forwards = np.ones((2, 5))
        correct_papw = np.array([[1, 1, 2, 2, 1],
                                 [1, 1, 2, 2, 1]])

        assert_allclose(self.learn_alg.B_forwards, correct_B_forwards)
        assert_allclose(self.learn_alg.papw, correct_papw)

    def test_get_influence_estimate(self):

        A, B = np.ones(2), np.ones((2, 5))
        self.learn_alg = Reverse_KF_RTRL(self.rnn, A=A, B=B)
        self.learn_alg.a_hat = np.array([1, 1, 2, 2, 1])
        self.rnn.a_J = np.eye(2)
        self.learn_alg.papw = np.array([[1, 1, 2, 2, 1],
                                        [1, 1, 2, 2, 1]])
        self.learn_alg.B_forwards = np.ones((2, 5))
        self.learn_alg.sample_nu = MagicMock()
        self.learn_alg.sample_nu.return_value = np.array([1, -1])
        A, B = self.learn_alg.get_influence_estimate()

        p0, p1 = np.sqrt(np.sqrt(5)), np.sqrt(np.sqrt(11))
        M_proj = np.array([[1, 1, 2, 2, 1],
                           [-1, -1, -2, -2, -1]])

        A_correct = p0 * np.ones(2) + p1 * np.array([1, -1])
        B_correct = (1 / p0) * np.ones((2, 5)) + (1 / p1) * M_proj
        assert_allclose(A, A_correct)
        assert_allclose(B, B_correct)

    def test_get_rec_grads(self):

        A, B = np.ones(2), np.ones((2, 5))
        self.learn_alg = Reverse_KF_RTRL(self.rnn, A=A, B=B)
        self.learn_alg.q = np.ones(2) * 0.5
        rec_grads = self.learn_alg.get_rec_grads()

        correct_rec_grads = np.ones((2, 5))

        assert_allclose(rec_grads, correct_rec_grads)

class Test_RFLO(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        cls.W_in = np.eye(2)
        cls.W_rec = np.eye(2)
        cls.W_out = np.eye(2)
        cls.b_rec = np.zeros(2)
        cls.b_out = np.zeros(2)

        cls.rnn = RNN(cls.W_in, cls.W_rec, cls.W_out,
                      cls.b_rec, cls.b_out,
                      activation=identity,
                      alpha=1,
                      output=softmax,
                      loss=softmax_cross_entropy)

        cls.rnn.a = np.ones(2)
        cls.rnn.a_prev = np.ones(2)
        cls.rnn.x = np.ones(2) * 2
        cls.rnn.error = np.ones(2) * 0.5

    def test_update_learning_vars(self):

        self.learn_alg = RFLO(self.rnn, alpha=0.5, B=np.ones((2, 5)))
        self.learn_alg.update_learning_vars()

        B_correct = np.array([[1, 1, 1.5, 1.5, 1],
                              [1, 1, 1.5, 1.5, 1]])

        assert_allclose(self.learn_alg.B, B_correct)

    def test_get_rec_grads(self):

        self.learn_alg = RFLO(self.rnn, alpha=0.5, B=np.ones((2, 5)))
        self.learn_alg.q = np.array([1, 2])
        rec_grads = self.learn_alg.get_rec_grads()

        correct_rec_grads = np.array([[1, 1, 1, 1, 1],
                                      [2, 2, 2, 2, 2]])

        assert_allclose(rec_grads, correct_rec_grads)

class Test_DNI(unittest.TestCase):

    pass

# class Test_Learning_Algorithm(unittest.TestCase):
#
#     @classmethod
#     def setUpClass(cls):
#         """Initializes task data and params so that simulations can be run."""
#
#         cls.task = Add_Task(4, 6, deterministic=True, tau_task=4)
#         cls.data = cls.task.gen_data(50, 50)
#
#         n_in = cls.task.n_in
#         n_h = 16
#         n_out = cls.task.n_out
#
#         cls.W_in = np.random.normal(0, np.sqrt(1/(n_in)), (n_h, n_in))
#         M_rand = np.random.normal(0, 1, (n_h, n_h))
#         cls.W_rec = np.linalg.qr(M_rand)[0]
#         cls.W_out = np.random.normal(0, np.sqrt(1/(n_h)), (n_out, n_h))
#         cls.W_FB = np.random.normal(0, np.sqrt(1/n_out), (n_out, n_h))
#
#         cls.b_rec = np.zeros(n_h)
#         cls.b_out = np.zeros(n_out)
#
#     def test_forward_bptt(self):
#         """Verifies that BPTT algorithm gives same aggregate weight change as
#         RTRL for a very small learning rate, while also checking that the
#         recurrent weights did change some amount (i.e. learning rate not *too*
#         small)."""
#
#         self.data = self.task.gen_data(400, 10)
#
#         alpha = 1
#
#         self.rnn_1 = RNN(self.W_in, self.W_rec, self.W_out,
#                          self.b_rec, self.b_out,
#                          activation=tanh,
#                          alpha=alpha,
#                          output=softmax,
#                          loss=softmax_cross_entropy)
#
#         self.rnn_2 = RNN(self.W_in, self.W_rec, self.W_out,
#                          self.b_rec, self.b_out,
#                          activation=tanh,
#                          alpha=alpha,
#                          output=softmax,
#                          loss=softmax_cross_entropy)
#
#         np.random.seed(1)
#         self.optimizer_1 = SGD(lr=0.000001)
#         self.learn_alg_1 = RTRL(self.rnn_1)
#         np.random.seed(1)
#         self.optimizer_2 = SGD(lr=0.000001)
#         self.learn_alg_2 = Forward_BPTT(self.rnn_2, 15)
#
#         monitors = []
#
#         np.random.seed(2)
#         self.sim_1 = Simulation(self.rnn_1)
#         self.sim_1.run(self.data, learn_alg=self.learn_alg_1,
#                        optimizer=self.optimizer_1,
#                        monitors=monitors,
#                        verbose=False)
#
#         np.random.seed(2)
#         self.sim_2 = Simulation(self.rnn_2)
#         self.sim_2.run(self.data, learn_alg=self.learn_alg_2,
#                        optimizer=self.optimizer_2,
#                        monitors=monitors,
#                        verbose=False)
#
#         #print(self.W_rec[0, 0], '/n', self.rnn_1.W_rec[0, 0], '/n', self.rnn_2.W_rec[0, 0])
#
#         #Assert networks learned similar weights with a small tolerance.
#         #assert_allclose(self.rnn_1.W_rec,
#         #                           self.rnn_2.W_rec, atol=1e-5))
#         #Assert networks' parameters changed appreciably, despite a large
#         #tolerance for closeness.
#         #self.assertFalse(assert_allclose(self.W_rec,
#         #                            self.rnn_2.W_rec, atol=1e-4))
#
#
#
#     def test_kernl_reduce_rflo(self):
#         """Verifies that KeRNL reduces to RFLO in special case.
#
#         If beta is initialized to the identity while the gammas are all
#         initialized to the network inverse time constant alpha, and the KeRNL
#         optimizer has 0 learning rate (i.e. beta and gamma do not change), then
#         KeRNL should produce the same gradients as RFLO if the approximate
#         KeRNL of (1 - alpha) (rather than exp(-alpha)) is used."""
#
#         self.data = self.task.gen_data(200, 100)
#
#         alpha = 0.3
#
#         self.rnn_1 = RNN(self.W_in, self.W_rec, self.W_out,
#                          self.b_rec, self.b_out,
#                          activation=tanh,
#                          alpha=alpha,
#                          output=softmax,
#                          loss=softmax_cross_entropy)
#
#         self.rnn_2 = RNN(self.W_in, self.W_rec, self.W_out,
#                          self.b_rec, self.b_out,
#                          activation=tanh,
#                          alpha=alpha,
#                          output=softmax,
#                          loss=softmax_cross_entropy)
#
#         #RFLO
#         np.random.seed(1)
#         self.optimizer_1 = SGD(lr=0.001)
#         self.learn_alg_1 = RFLO(self.rnn_1, alpha)
#         #KeRNL with beta and gamma fixed to RFLO values
#         np.random.seed(1)
#         self.optimizer_2 = SGD(lr=0.001)
#         self.KeRNL_optimizer = SGD(lr=0)
#         beta = np.eye(self.rnn_2.n_h)
#         gamma = np.ones(self.rnn_2.n_h)*alpha
#         self.learn_alg_2 = KeRNL(self.rnn_2, self.KeRNL_optimizer,
#                                  beta=beta, gamma=gamma,
#                                  use_approx_kernel=True)
#
#         monitors = []
#
#         np.random.seed(2)
#         self.sim_1 = Simulation(self.rnn_1)
#         self.sim_1.run(self.data, learn_alg=self.learn_alg_1,
#                        optimizer=self.optimizer_1,
#                        monitors=monitors,
#                        verbose=False)
#
#         np.random.seed(2)
#         self.sim_2 = Simulation(self.rnn_2)
#         self.sim_2.run(self.data, learn_alg=self.learn_alg_2,
#                        optimizer=self.optimizer_2,
#                        monitors=monitors,
#                        verbose=False)
#
#         #Assert networks learned similar weights with a small tolerance.
#         assert_allclose(self.rnn_1.W_rec,
#                                    self.rnn_2.W_rec, atol=1e-5))
#         #Assert networks' parameters changed appreciably, despite a large
#         #tolerance for closeness.
#         self.assertFalse(assert_allclose(self.W_rec,
#                                     self.rnn_2.W_rec, atol=1e-3))

if __name__ == '__main__':
    unittest.main()
