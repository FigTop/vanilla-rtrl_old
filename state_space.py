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

    def __init__(self, trajectories, n_PCs=2, add_fig=True):
        """The array trajectories must have a shape of (sample, unit)"""

        self.n_PCs = n_PCs

        self.add_fig = add_fig

        self.trajectories = trajectories

        self.U, self.S, self.V = np.linalg.svd(self.trajectories)

        if self.add_fig:
            self.fig = plt.figure()
            if self.n_PCs == 2:
                self.ax = self.fig.add_subplot(111)
            if self.n_PCs == 3:
                self.ax = self.fig.add_subplot(111, projection='3d')

    def plot_in_state_space(self, trajectories, *args, **kwargs):

        PCs = (self.V.T.dot(trajectories.T)).T[:, :self.n_PCs]

        if self.n_PCs == 2:
            self.ax.plot(PCs[:, 0], PCs[:, 1], *args, **kwargs)
            self.ax.plot([PCs[0, 0]], [PCs[0, 1]], '*', color=kwargs['color'])
        if self.n_PCs == 3:
            self.ax.plot(PCs[:, 0], PCs[:, 1], PCs[:, 2], *args, **kwargs)

    def clear_plot(self):

        self.fig.axes[0].clear()

if __name__ == '__main__':
    ssa2 = State_Space_Analysis(test_sim.mons['net.a'], n_PCs=3)
    a = test_sim.mons['net.a'].reshape((-1, task.time_steps_per_trial, n_hidden))
    x = data['test']['X'].reshape((-1, task.time_steps_per_trial, n_in))
    on_trials = np.where(x[:, 1, 0] > 0)[0]
    for i in range(a.shape[0]):
        if i in on_trials:
            ssa2.plot_in_state_space(a[i], color='b', alpha=0.2)
            ssa2.plot_in_state_space(a[i, :1], '.', color='b', alpha=0.6)
        else:
            ssa2.plot_in_state_space(a[i], color='g', alpha=0.2)
            ssa2.plot_in_state_space(a[i, :1], '.', color='g', alpha=0.6)