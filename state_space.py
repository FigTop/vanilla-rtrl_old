#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 17:18:46 2019
@author: omarschall
"""

import numpy as np
import matplotlib.pyplot as plt
from utils import *
from dynamics import *
from mpl_toolkits.mplot3d import Axes3D


class State_Space_Analysis:

    def __init__(self, checkpoint, test_data, dim_reduction_method=Vanilla_PCA,
                 V=None, **kwargs):
        """The array trajectories must have a shape of (sample, unit)"""

        if V is None:
            self.V = dim_reduction_method(checkpoint, test_data, **kwargs)
        else:
            self.V = V

        self.dim = self.V.shape[1]

        self.fig = plt.figure()
        if self.dim == 2:
            self.ax = self.fig.add_subplot(111)
        if self.dim == 3:
            self.ax = self.fig.add_subplot(111, projection='3d')

    def plot_in_state_space(self, trajectories, mark_start_and_end=True,
                            color='C0', *args, **kwargs):
        """Plots given trajectories' projection onto axes as defined in
        __init__ by training data."""

        projs = (self.V.T.dot(trajectories.T)).T

        if self.dim == 2:
            self.ax.plot(projs[:, 0], projs[:, 1], *args, **kwargs,
                         color=color)
            if mark_start_and_end:
                self.ax.plot([projs[0, 0]], [projs[0, 1]], '*', color=color)
                self.ax.plot([projs[-1, 0]], [projs[-1, 1]], 'x', color=color)
        if self.dim == 3:
            self.ax.plot(projs[:, 0], projs[:, 1], projs[:, 2],
                         *args, **kwargs, color=color)
            if mark_start_and_end:
                self.ax.plot([projs[0, 0]], [projs[0, 1]], [projs[0, 2]],
                             '*', color=color)
                self.ax.plot([projs[-1, 0]], [projs[-1, 1]], [projs[-1, 1]],
                             'x', color=color)

    def clear_plot(self):
        """Clears all plots from figure"""

        self.fig.axes[0].clear()