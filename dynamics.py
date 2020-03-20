#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 13:48:33 2020

@author: omarschall
"""

import numpy as np
from simulation import *

def run_autonomous_sim(rnn, N, a_initial, monitors=[]):
    """Creates and runs a test simulation with no inputs and a specified
    initial state of the network."""
    
    #Create empty data array
    data = {'test': {'X': np.zeros((N, rnn.n_in)),
                     'Y': np.zeros((N, rnn.n_out))}}
    sim = Simulation(rnn)
    sim.run(data, mode='test', monitors=monitors,
            a_initial=a_initial,
            verbose=True,
            check_accuracy=False,
            check_loss=False)
    
    return sim