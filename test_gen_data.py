#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 14:19:07 2019

@author: omarschall
"""

import numpy as np
import unittest
from gen_data import *

class Test_Network(unittest.TestCase):
    """Tests methods from the network.py module."""

    @classmethod
    def setUpClass(cls):
        """Initializes a simple instance of network for testing."""
        
        pass
    
    def test_coin_task(self):
        
        task = Coin_Task(6, 10, one_hot=True, deterministic=True, tau_task=1)
        data = task.gen_data(50, 0)
        
        for i in range(12, 25):
            y = (0.5 +
                 0.5*data['train']['X'][i-5, 0] -
                 0.25*data['train']['X'][i-9, 0])
            self.assertEqual(data['train']['Y'][i, 0], y)
            
        task = Coin_Task(6, 10, one_hot=True, deterministic=True, tau_task=2)
        data = task.gen_data(50, 0)
        
        for i in range(25, 35):
            if i%2 == 1:
                x1 = data['train']['X'][i, 0]
                x2 = data['train']['X'][i-1, 0]
                self.assertEqual(x1, x2)
                y = (0.5 +
                     0.5*data['train']['X'][i-11, 0] -
                     0.25*data['train']['X'][i-19, 0])
                self.assertEqual(data['train']['Y'][i, 0], y)
            if i%2 == 0:
                x1 = data['train']['X'][i, 0]
                x2 = data['train']['X'][i+1, 0]
                self.assertEqual(x1, x2)
                y = (0.5 +
                     0.5*data['train']['X'][i-10, 0] -
                     0.25*data['train']['X'][i-18, 0])
                self.assertEqual(data['train']['Y'][i, 0], y)
            
    def test_copy_task(self):
        
        task = Copy_Task(3, 10)
        data = task.gen_data(100, 0)
        
        data['train']['X']
        

if __name__=='__main__':
    unittest.main()