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
        data = task.gen_data(100, 0)
        
        for i in range(12, 25):
            y = (0.5 +
                 0.5*data['train']['X'][i-5, 0] -
                 0.25*data['train']['X'][i-9, 0])
            self.assertEqual(data['train']['Y'][i, 0], y)
            
    def test_copy_task(self):
        
        task = Copy_Task(3, 10)
        data = task.gen_data(100, 0)
        
        data['train']['X'] = 
        

if __name__=='__main__':
    unittest.main()