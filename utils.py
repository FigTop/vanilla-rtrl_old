#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  3 12:08:30 2019

@author: omarschall
"""

import matplotlib.pyplot as plt
import numpy as np
import itertools
from scipy.stats import unitary_group

### --- Mathematical tools --- ###

def norm(z):
    """Computes the L2 norm of a numpy array."""

    return np.sqrt(np.sum(np.square(z)))

def split_weight_matrix(A, sizes, axis=1):
    """Splits a weight matrix along the specified axis (0 for row, 1 for
    column) into a list of sub arrays of size specified by 'sizes'."""

    idx = [0] + np.cumsum(sizes).tolist()
    if axis == 1:
        ret = [np.squeeze(A[:,idx[i]:idx[i+1]]) for i in range(len(idx) - 1)]
    elif axis == 0:
        ret = [np.squeeze(A[idx[i]:idx[i+1],:]) for i in range(len(idx) - 1)]
    return ret

def rectangular_filter(signal, filter_size=100):
    """Convolves a given signal with a rectangular filter in 'valid' mode

    Args:
        signal (numpy array): An 1-dimensional array specifying the signal.
        filter_size (int): An integer specifcying the width of the rectangular
            filter used for the convolution."""

    return np.convolve(signal, np.ones(filter_size)/filter_size, mode='valid')

def classification_accuracy(data, y_hat):
    """Calculates the fraction of test data whose argmax matches that of
    the prediction."""

    y_hat = np.array(y_hat)

    i_label = np.argmax(data['test']['Y'], axis=1)
    i_pred = np.argmax(y_hat, axis=1)

    acc = np.sum(i_label==i_pred)/len(i_label)

    return acc

def normalized_dot_product(a, b):
    """Calculates the normalized dot product between two numpy arrays, after
    flattening them."""

    a_norm = norm(a)
    b_norm = norm(b)
    
    if a_norm >0 and b_norm > 0:
        return np.dot(a.flatten(),b.flatten())/(a_norm*b_norm)
    else:
        return 0

def get_spectral_radius(M):
    """Calculates the spectral radius of a matrix."""

    eigs, _ = np.linalg.eig(M)

    return np.amax(np.absolute(eigs))

def generate_real_matrix_with_given_eigenvalues(evals):
    """For a given set of complex eigenvalues, generates a real matrix with
    those eigenvalues.

    More precisely, the user should specify *half* of the eigenvalues (since
    the other half must be complex conjugates). Thus the dimension should be
    even for conveneince.

    Args:
        evals (numpy array): An array of shape (n_half), where n_half is half
            the dimensionality of the matrix to be generated, that specifies
            the desired eigenvalues.

    Returns:
        A real matrix, half of whose eigenvalues are evals."""

    n_half = len(evals)
    evals = np.concatenate([evals, np.conjugate(evals)])
    
    evecs = unitary_group.rvs(2*n_half)[:,:n_half]
    evecs = np.concatenate([evecs, np.conjugate(evecs)], axis=1)

    M = evecs.dot(np.diag(evals)).dot(evecs)

    return np.real(M)

def plot_eigenvalues(M, fig=None, return_fig=False):
    """Plots eigenvalues of a given matrix in the complex plane, as well
    as the unit circle for reference."""

    eigs, _ = np.linalg.eig(M)

    if fig is None:
        fig = plt.figure()
    plt.plot(np.real(eigs), np.imag(eigs), '.')
    theta = np.arange(0, 2*np.pi, 0.01)
    plt.plot(np.cos(theta), np.sin(theta), 'k')
    plt.axis('equal')

    if return_fig:
        return fig

### --- Programming tools --- ###

def config_generator(**kwargs):
    """Generator object that produces a Cartesian product of configurations.

    Each kwarg should be a list of possible values for the key. Yields a
    dictionary specifying a particular configuration."""

    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))