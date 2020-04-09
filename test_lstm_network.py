#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import unittest
from lstm_network import *

class Test_Network(unittest.TestCase):
    """Tests methods from the network.py module."""

    @classmethod
    def setUpClass(cls):
        """Initializes a simple instance of network for testing."""
        n_in = 2
        n_h = 2
        n_t = 4
        n_out = 2
        n_h_hat = 4

        W_f = np.concatenate([np.eye(n_h),np.ones((n_in,n_in))],axis=1)
        W_i = np.concatenate([np.eye(n_h),np.ones((n_in,n_in))],axis=1)
        W_a = np.concatenate([np.eye(n_h),np.ones((n_in,n_in))],axis=1)
        W_o = np.concatenate([np.eye(n_h),np.ones((n_in,n_in))],axis=1)
        W_c_out = np.eye(n_out)
        W_h_out = np.eye(n_out)

        b_i = np.zeros(n_h)
        b_f = np.zeros(n_h)
        b_a = np.zeros(n_h)
        b_o = np.zeros(n_h)

        b_out = np.zeros(n_out)

        cls.lstm = LSTM(W_f, W_i, W_a, W_o, W_c_out, W_h_out,
                 b_f, b_i, b_a, b_o, b_out,
                 output = softmax, loss = softmax_cross_entropy)


    def test_reset_network(self):
        
        np.random.seed(1)
        h = np.random.normal(0, 1, (self.lstm.n_h))
        c = np.random.normal(0, 1, (self.lstm.n_h))

        self.lstm.reset_network(h=h,c=c)
        self.assertTrue(np.isclose(self.lstm.prev_h, h).all())
        self.assertTrue(np.isclose(self.lstm.prev_c, c).all())
        z1 = np.copy(self.lstm.z)

        np.random.seed(1)
        self.lstm.reset_network(sigma=1)
        z2 = np.copy(self.lstm.z)
        self.assertTrue(np.isclose(z1, z2).all())

    def test_next_state(self):
        h = np.ones(2)
        c = np.array([1,2])
        x = np.ones(2)
        self.lstm.reset_network(h=h,c=c)
        self.lstm.next_state(x)

        f = sigmoid.f(np.array([3,3]))
        i = sigmoid.f(np.array([3,3]))
        a = tanh.f(np.array([3,3]))
        c_next = a * i + f * c
        o = sigmoid.f(np.array([3,3]))
        h_next = tanh.f(c_next)*o

        self.assertTrue(np.isclose(self.lstm.h, h_next).all())
        self.assertTrue(np.isclose(self.lstm.h_prev, h).all())

    def test_z_out(self):
        h = np.ones(2)
        c = np.array([1,2])
        self.lstm.reset_network(h=h,c=c)
        self.lstm.z_out()
        z= np.array([2,3])
        self.assertTrue(np.isclose(self.lstm.z, z).all())

    def test_get_a_jacobian(self):
        h = np.ones(2)
        c = np.array([1,2])
        self.lstm.x = np.ones(2)
        self.lstm.reset_network(h=h,c=c)
        self.lstm.get_a_jacobian()

        
        f = sigmoid.f(np.array([3,3]))
        i = sigmoid.f(np.array([3,3]))
        a = tanh.f(np.array([3,3]))
        c_next = a * i + f * c
        o = sigmoid.f(np.array([3,3]))
        h_next = tanh.f(c_next)*o

        c_c_J = np.diag(f)
        c_c_J_c = self.lstm.a_J[:2,:2]
        self.assertTrue(np.isclose(c_c_J_c, c_c_J).all())

        P_1 = tanh.f_prime(np.array([3,3]))
        P_2 = sigmoid.f_prime(np.array([3,3]))

        c_h_J = np.diag(i*P_1)+ np.diag(a*P_2)+ np.diag(c*P_2)
        c_h_J_c = self.lstm.a_J[:2,2:]
        self.assertTrue(np.isclose(c_h_J_c, c_h_J).all())

        h_c_J = np.diag(f*o*(1-tanh.f(c_next)**2))
        h_c_J_c = self.lstm.a_J[2:,:2]
        self.assertTrue(np.isclose(h_c_J_c, h_c_J).all())  

        h_h_J = np.diag(o * tanh.f_prime(c_next)*(i*P_1+a*P_2+c*P_2))+np.diag(tanh.f(c_next)*(o-o**2))
        h_h_J_c = self.lstm.a_J[2:,2:]
        self.assertTrue(np.isclose(h_h_J_c, h_h_J).all())  
                

# %%
if __name__=='__main__':
    unittest.main()


