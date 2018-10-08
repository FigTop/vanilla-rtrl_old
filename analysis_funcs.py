#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 12:02:11 2018

@author: omarschall
"""

import numpy as np
import matplotlib.pyplot as plt

def plot_smoothed_loss(mons, filter_size=100):
    
    fig = plt.figure()
    
    losses = mons['loss_']
    smoothed_loss = np.convolve(losses, np.ones(filter_size)/filter_size, mode='valid')
    
    plt.plot(smoothed_loss)
    plt.plot([0, len(smoothed_loss)], [0.66, 0.66], '--', color='r')
    plt.plot([0, len(smoothed_loss)], [0.52, 0.52], '--', color='m')
    plt.plot([0, len(smoothed_loss)], [0.45, 0.45], '--', color='g')
    
    plt.ylim([0,1])
    
    return fig

def get_spectral_radius(M):
    
    eigs, _ = np.linalg.eig(M)
    
    return np.amax(np.absolute(eigs))

def get_spectral_radii(Ms):
    
    r = []
    for i in range(Ms.shape[0]):
        
        r.append(get_spectral_radius(Ms[i,:,:]))
        
    return r

def get_vector_alignment(v1, v2):

    alignment = []
    for i in range(v1.shape[0]):
        
        a = v1[i,:].flatten()
        b = v2[i,:].flatten()
        
        alignment.append(np.dot(a,b)/np.sqrt(np.sum(a**2)*np.sum(b**2)))
        
    return alignment
    
def plot_filtered_signals(signals, filter_size=100, y_lim=[0,1.5], plot_loss_benchmarks=True):
    
    fig = plt.figure(figsize=[8, 6])
    
    for signal in signals:
        smoothed_signal = np.convolve(signal, np.ones(filter_size)/filter_size, mode='valid')
        plt.plot(smoothed_signal)
    
    if plot_loss_benchmarks:
        plt.plot([0, len(smoothed_signal)], [0.66, 0.66], '--', color='r')
        plt.plot([0, len(smoothed_signal)], [0.52, 0.52], '--', color='m')
        plt.plot([0, len(smoothed_signal)], [0.45, 0.45], '--', color='g')
    
    plt.ylim(y_lim)
    plt.xlabel('Time')
    
    return fig













