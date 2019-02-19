#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 17:18:46 2019

@author: omarschall
"""

import numpy as np
import matplotlib.pyplot as plt
from utils import *
from mpl_toolkits.mplot3d import Axes3D

class State_Space_Analysis:
    
    def __init__(self, trajectories, n_PCs=2):
        '''
        The array trajectories must have a shape of (sample, unit)
        '''
        
        self.n_PCs = n_PCs
        self.fig = plt.figure()
        if n_PCs==2:
            self.ax = self.fig.add_subplot(111)
        if n_PCs==3:
            self.ax = self.fig.add_subplot(111, projection='3d')
        
        self.trajectories = trajectories
        
        self.U, self.S, self.V = np.linalg.svd(self.trajectories)
        
    def plot_in_state_space(self, trajectories, *args, **kwargs):
        
        PCs = (self.V.T.dot(trajectories.T)).T[:,:self.n_PCs]
        
        
        if self.n_PCs==2:
            self.ax.plot(PCs[:,0], PCs[:,1], *args, **kwargs)
        if self.n_PCs==3:
            self.ax.plot(PCs[:,0], PCs[:,1], PCs[:,2], *args, **kwargs)