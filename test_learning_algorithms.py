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
        
        cls.task = Coin_Task(4, 6, one_hot=True, deterministic=True, tau_task=4)
        #task = Sine_Wave(0.001, [0.01, 0.007, 0.003, 0.001], amplitude=0.1, method='regular')
        cls.data = cls.task.gen_data(100, 100)
        
        n_in     = cls.task.n_in
        n_hidden = 32
        n_out    = cls.task.n_out
        
        cls.W_in  = np.random.normal(0, np.sqrt(1/(n_in)), (n_hidden, n_in))
        cls.W_rec = np.linalg.qr(np.random.normal(0, 1, (n_hidden, n_hidden)))[0]
        cls.W_out = np.random.normal(0, np.sqrt(1/(n_hidden)), (n_out, n_hidden))
        cls.W_FB = np.random.normal(0, np.sqrt(1/n_out), (n_out, n_hidden))
        
        cls.b_rec = np.zeros(n_hidden)
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
        
        np.random.seed(1)
        self.optimizer_1 = SGD(lr=0.0005)
        self.SG_optimizer_1 = SGD(lr=0.01)
        self.learn_alg_1 = DNI(self.rnn_1, self.SG_optimizer_1, W_a_lr=0.01, backprop_weights='approximate',
                               SG_label_activation=tanh, W_FB=self.W_FB)
        np.random.seed(1)
        self.optimizer_2 = SGD(lr=0.0005)
        self.SG_optimizer_2 = SGD(lr=0.01)
        self.learn_alg_2 = DNI(self.rnn_2, self.SG_optimizer_2, W_a_lr=0.01, backprop_weights='approximate',
                               SG_label_activation=tanh, W_FB=self.W_FB, alpha_e=1)
        monitors = ['loss_', 'y_hat', 'a']
        
        np.random.seed(2)
        self.sim_1 = Simulation(self.rnn_1, self.learn_alg_1, self.optimizer_1, L2_reg=0.0001)
        self.sim_1.run(self.data,
                       monitors=monitors,
                       verbose=False)
        
        np.random.seed(2)
        self.sim_2 = Simulation(self.rnn_2, self.learn_alg_2, self.optimizer_2, L2_reg=0.0001)
        self.sim_2.run(self.data,
                       monitors=monitors,
                       verbose=False)
        
        self.assertTrue(np.isclose(self.sim_1.mons['a'], self.sim_2.mons['a']).all())
        
    def test_forward_bptt(self):
        
        self.data = self.task.gen_data(2000, 100)
        
        np.random.seed(1)
        self.optimizer_1 = SGD(lr=0.000001)
        self.learn_alg_1 = RTRL(self.rnn_1)
        np.random.seed(1)
        self.optimizer_2 = SGD(lr=0.000001)
        self.learn_alg_2 = Forward_BPTT(self.rnn_2, 15)
        
        monitors = ['loss_', 'y_hat', 'a']
        
        np.random.seed(2)
        self.sim_1 = Simulation(self.rnn_1, self.learn_alg_1, self.optimizer_1, L2_reg=0.0001)
        self.sim_1.run(self.data,
                       monitors=monitors,
                       verbose=False)
        
        np.random.seed(2)
        self.sim_2 = Simulation(self.rnn_2, self.learn_alg_2, self.optimizer_2, L2_reg=0.0001)
        self.sim_2.run(self.data,
                       monitors=monitors,
                       verbose=False)
        
        self.assertTrue(np.isclose(self.rnn_1.W_rec, self.rnn_2.W_rec, atol=1e-5).all())
        self.assertFalse(np.isclose(self.W_rec, self.rnn_2.W_rec, atol=1e-3).all())
        
    def test_kernl_reduce_rflo(self):
        
        self.data = self.task.gen_data(2000, 100)
        
        alpha = 0.3
        
        self.rnn_1.alpha = alpha
        self.rnn_2.alpha = alpha
        
        np.random.seed(1)
        self.optimizer_1 = SGD(lr=0.001)
        self.learn_alg_1 = RFLO(self.rnn_1, alpha)
        np.random.seed(1)
        self.optimizer_2 = SGD(lr=0.001)
        self.KeRNL_optimizer = SGD(lr=0)
        beta = np.eye(self.rnn_2.n_hidden)
        gamma = np.ones(self.rnn_2.n_hidden)*alpha
        self.learn_alg_2 = KeRNL(self.rnn_2, self.KeRNL_optimizer,
                                 beta=beta, gamma=gamma,
                                 use_approx_kernel=True)
        
        monitors = ['loss_', 'y_hat', 'a']
        
        np.random.seed(2)
        self.sim_1 = Simulation(self.rnn_1, self.learn_alg_1, self.optimizer_1, L2_reg=0.0001)
        self.sim_1.run(self.data,
                       monitors=monitors,
                       verbose=False)
        
        np.random.seed(2)
        self.sim_2 = Simulation(self.rnn_2, self.learn_alg_2, self.optimizer_2, L2_reg=0.0001)
        self.sim_2.run(self.data,
                       monitors=monitors,
                       verbose=False)
        
        self.assertTrue(np.isclose(self.rnn_1.W_rec, self.rnn_2.W_rec, atol=1e-5).all())
        self.assertFalse(np.isclose(self.W_rec, self.rnn_2.W_rec, atol=1e-3).all())
        
if __name__=='__main__':
    unittest.main()