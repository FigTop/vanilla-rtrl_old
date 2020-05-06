#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 17:18:46 2019
@author: omarschall
"""

import numpy as np
try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    pass
from utils import *
from dynamics import *
from mpl_toolkits.mplot3d import Axes3D


class State_Space_Analysis:

    def __init__(self, checkpoint, test_data, dim_reduction_method=Vanilla_PCA,
                 transform=None, **kwargs):
        """The array trajectories must have a shape of (sample, unit)"""

        if transform is None:
            self.transform = dim_reduction_method(checkpoint, test_data,
                                                  **kwargs)
        else:
            self.transform = transform

        dummy_data = np.zeros((10, checkpoint['rnn'].n_h))
        self.dim = self.transform(dummy_data).shape[1]

        self.fig = plt.figure()
        if self.dim == 2:
            self.ax = self.fig.add_subplot(111)
        if self.dim == 3:
            self.ax = self.fig.add_subplot(111, projection='3d')

    def plot_in_state_space(self, trajectories, mark_start_and_end=False,
                            color='C0', *args, **kwargs):
        """Plots given trajectories' projection onto axes as defined in
        __init__ by training data."""

        projs = self.transform(trajectories)

        if self.dim == 2:
            self.ax.plot(projs[:, 0], projs[:, 1], *args, **kwargs,
                         color=color)
            if mark_start_and_end:
                self.ax.plot([projs[0, 0]], [projs[0, 1]], 'x', color=color)
                self.ax.plot([projs[-1, 0]], [projs[-1, 1]], 'o', color=color)
        if self.dim == 3:
            self.ax.plot(projs[:, 0], projs[:, 1], projs[:, 2],
                         *args, **kwargs, color=color)
            if mark_start_and_end:
                self.ax.plot([projs[0, 0]], [projs[0, 1]], [projs[0, 2]],
                             'x', color=color)
                self.ax.plot([projs[-1, 0]], [projs[-1, 1]], [projs[-1, 2]],
                             'o', color=color)

    def clear_plot(self):
        """Clears all plots from figure"""

        self.fig.axes[0].clear()

def plot_checkpoint_results(checkpoint, data, ssa=None, plot_test_points=False,
                            plot_fixed_points=False, plot_cluster_means=False,
                            plot_uncategorized_points=False,
                            plot_init_points=False, eig_norm_color=False,
                            plot_graph_structure=False,
                            plot_vae_sample=False,
                            plot_test_sample=False):

    rnn = checkpoint['rnn']
    test_sim = Simulation(rnn)
    test_sim.run(data,
                  mode='test',
                  monitors=['rnn.loss_', 'rnn.y_hat', 'rnn.a'],
                  verbose=False)

    A_init = checkpoint['A_init']
    fixed_points = checkpoint['fixed_points']
    labels = checkpoint['cluster_labels']
    cluster_means = checkpoint['cluster_means']
    cluster_eigs = checkpoint['cluster_eigs']

    if ssa is None:
        transform = partial(np.dot, b=checkpoint['V'])
        ssa = State_Space_Analysis(checkpoint, data, transform=transform)
    ssa.clear_plot()
    if plot_test_points:
        ssa.plot_in_state_space(test_sim.mons['rnn.a'][1000:], False, 'C0',
                                '.', alpha=0.009)
    if plot_test_sample:
        T = checkpoint['VAE_T']
        T_total = test_sim.mons['rnn.a'].shape[0]
        t_start = np.random.randint(0, T_total - T)
        ssa.plot_in_state_space(test_sim.mons['rnn.a'][t_start:t_start + T],
                                False, 'C0', alpha=0.7)
        plt.figure()
        plt.plot(data['test']['X'][:, 0] + 2.5, (str(0.6)), linestyle='--')
        plt.plot(data['test']['Y'][:, 0] + 2.5, 'C0')
        plt.plot(test_sim.mons['rnn.y_hat'][:, 0] + 2.5, 'C3')
        plt.plot(data['test']['X'][:, 1], (str(0.6)), linestyle='--')
        plt.plot(data['test']['Y'][:, 1], 'C0')
        plt.plot(test_sim.mons['rnn.y_hat'][:, 1], 'C3')
        plt.plot(data['test']['X'][:, 2] - 2.5, (str(0.6)), linestyle='--')
        plt.plot(data['test']['Y'][:, 2] - 2.5, 'C0')
        plt.plot(test_sim.mons['rnn.y_hat'][:, 2] - 2.5, 'C3')
        plt.xlim([t_start, t_start + T])
        plt.yticks([])
        plt.xlabel('time steps')

    if plot_init_points:
        ssa.plot_in_state_space(A_init, False, 'C9', 'x', alpha=1)

    cluster_idx = np.unique(labels)
    n_clusters = len(cluster_idx) - (-1 in cluster_idx)
    for i in cluster_idx:

        if i == -1:
            color = 'k'
            if not plot_uncategorized_points:
                continue
        else:
            color = 'C{}'.format(i+1)
        if plot_fixed_points:
            ssa.plot_in_state_space(fixed_points[labels == i], False, color, '*', alpha=0.5)

    if plot_cluster_means:
        if eig_norm_color:
            ssa.plot_in_state_space(cluster_means[cluster_eigs<1], False, 'k', 'X', alpha=0.3)
            ssa.plot_in_state_space(cluster_means[cluster_eigs>1], False, 'k', 'o', alpha=0.3)
        else:
            ssa.plot_in_state_space(cluster_means, False, 'k', 'X', alpha=0.3)

    if plot_graph_structure:

        graph = checkpoint['adjacency_matrix']
        for i, j in zip(*np.where(graph != 0)):

            if i == j:
                continue

            weight = graph[i, j]
            line = np.array([cluster_means[i], cluster_means[j]])
            ssa.plot_in_state_space(line, True, color='k', alpha=weight)

    if plot_vae_sample:

        traj = sample_from_VAE(checkpoint)
        ssa.plot_in_state_space(traj, True, 'C3', alpha=0.7)

    return ssa