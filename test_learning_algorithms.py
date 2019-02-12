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

class Test_Learning_Algorithm(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        
        task = Coin_Task(4, 6, one_hot=True, deterministic=True, tau_task=4)
        #task = Sine_Wave(0.001, [0.01, 0.007, 0.003, 0.001], amplitude=0.1, method='regular')
        cls.data = task.gen_data(100, 100)
        
        n_in     = task.n_in
        n_hidden = 32
        n_out    = task.n_out
        
        W_in  = np.random.normal(0, np.sqrt(1/(n_in)), (n_hidden, n_in))
        W_rec = np.linalg.qr(np.random.normal(0, 1, (n_hidden, n_hidden)))[0]
        W_out = np.random.normal(0, np.sqrt(1/(n_hidden)), (n_out, n_hidden))
        cls.W_FB = np.random.normal(0, np.sqrt(1/n_out), (n_out, n_hidden))
        
        b_rec = np.zeros(n_hidden)
        b_out = np.zeros(n_out)
        
        alpha = 1
        
        cls.rnn_1 = RNN(W_in, W_rec, W_out, b_rec, b_out,
                        activation=tanh,
                        alpha=alpha,
                        output=softmax,
                        loss=softmax_cross_entropy)
        
        cls.rnn_2 = RNN(W_in, W_rec, W_out, b_rec, b_out,
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
        
if __name__=='__main__':
    unittest.main()