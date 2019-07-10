#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 16:01:00 2019

@author: omarschall
"""

import unittest
from optimizers import *
import numpy as np

class Test_SGD(unittest.TestCase):

    @classmethod
    def setUp(cls):

        cls.optimizer = SGD(lr=0.1, clip_norm=2)

    def test_clip_norm(self):

        grads = [np.ones(2)*2, np.ones(2)]
        clipped_grads = self.optimizer.clip_gradient((grads))
        grad_norm = np.sqrt(10)
        correct_clipped_grads = [np.ones(2)*4/grad_norm,
                                 np.ones(2)*2/grad_norm]

        self.assertTrue(np.isclose(clipped_grads,
                                   correct_clipped_grads).all())

    def test_update(self):

        params = [np.ones(2)]
        grads = [np.ones(2)]
        updated_params = self.optimizer.get_updated_params(params, grads)
        correct_updated_params = [np.ones(2)*0.9]

        self.assertTrue(np.isclose(updated_params,
                                   correct_updated_params).all())


if __name__ == '__main__':
    unittest.main()
