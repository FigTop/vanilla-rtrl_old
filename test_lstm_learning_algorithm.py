#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# %%
import numpy as np
from numpy.testing import assert_allclose
import unittest
from unittest.mock import MagicMock
from lstm_network import LSTM
from simulation import Simulation
from lstm_learning_algorithms import *
from functions import *
from gen_data import *
from optimizers import *
from pdb import set_trace

class Test_LSTM_Learning_Algorithm(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        cls.W_f = np.concatenate([np.eye(2),np.eye(2)],axis=1)
        cls.W_i = np.concatenate([np.eye(2),np.eye(2)],axis=1)
        cls.W_a = np.concatenate([np.eye(2),np.eye(2)],axis=1)
        cls.W_o = np.concatenate([np.eye(2),np.eye(2)],axis=1)
        cls.b_f = np.zeros(2)
        cls.b_i = np.zeros(2)
        cls.b_a = np.zeros(2)
        cls.b_o = np.zeros(2)
        cls.W_FB = -np.ones((2, 2)) + np.eye(2)

        cls.W_c_out = np.eye(2)
        cls.W_h_out = np.eye(2)
        cls.b_out = np.zeros(2)

        cls.lstm = LSTM(cls.W_f, cls.W_i, cls.W_a, cls.W_o, cls.W_c_out, cls.W_h_out,
                   cls.b_f, cls.b_i, cls.b_a, cls.b_o, cls.b_out,
                   output=softmax,
                   loss=softmax_cross_entropy)

        cls.lstm.h = np.ones(2)
        cls.lstm.c = np.ones(2)

        cls.lstm.error = np.ones(2) * 0.5
        cls.lstm.x = np.ones(2)

    def test_update_learning_vars(self):

        self.learn_alg = RTRL_LSTM(self.lstm)
        self.learn_alg.update_learning_vars(self)






    def test_get_outer_grads(self):

        self.learn_alg = RTRL_LSTM(self.lstm)
        outer_grads = self.learn_alg.get_outer_grads()
        correct_outer_grads = np.ones((2,5)) * 0.5
        assert_allclose(outer_grads,correct_outer_grads)
        

    def test_propagate_feedback_to_hidden(self):

        #Case with symmetric feedback
        self.learn_alg = RTRL_LSTM(self.lstm)
        self.learn_alg.propagate_feedback_to_hidden()
        correct_q = np.ones(4) * 0.5
        assert_allclose(self.learn_alg.q, correct_q)

        #Case with random feedback
        # self.learn_alg = Learning_Algorithm(self.lstm, W_FB=self.W_FB)
        # self.learn_alg.propagate_feedback_to_hidden()
        # correct_q = -np.ones(4) * 0.5
        # assert_allclose(self.learn_alg.q, correct_q)

if __name__ == '__main__': 
    unittest.main()



