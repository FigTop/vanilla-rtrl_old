#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 23:02:35 2018

@author: omarschall
"""

import numpy as np
from network import RNN
from utils import *
from gen_data import gen_data
import matplotlib.pyplot as plt
import time
from optimizers import *

As = np.array(rnn.mons['A'])
Ws = np.array(rnn.mons['W_rec'])
grads = rnn.mons['grads']
losses = np.array(rnn.mons['loss_'])
u = np.array(rnn.mons['u'])
a_t = np.array(rnn.mons['a'])
h = np.array(rnn.mons['h'])

W_rec_grads = np.array([grad[0] for grad in grads])

plt.figure()
smoothed_loss = np.convolve(losses, np.ones(100)/100, mode='valid')
plt.plot(smoothed_loss)
plt.plot([0, len(smoothed_loss)], [0.66, 0.66], '--', color='r')
plt.plot([0, len(smoothed_loss)], [0.52, 0.52], '--', color='m')
plt.plot([0, len(smoothed_loss)], [0.45, 0.45], '--', color='g')
plt.ylim([0,1])
#plt.plot(losses, '.', alpha=0.4)



plt.figure()
dAdt = As[1:,:,:] - As[:-1,:,:]
dWdt = Ws[1:,:,:] = Ws[:-1,:,:]
alignment = []
for i in range(dAdt.shape[0]):
    
    #a = u[i,:32]
    #w = a_t[i,:]
    #w = 
    a = dAdt[i,:,:].flatten()
    w = dWdt[i,:,:].flatten()
    #a = As[i,:,:].flatten()
    #w = W_rec_grads[i,:,:].flatten()
    
    alignment.append(np.dot(a,w)/np.sqrt(np.sum(a**2)*np.sum(w**2)))
    
plt.plot(alignment, '.', alpha=0.2)


plt.figure()
for i, col in zip([0, 10, 100, 2500, 10000, 25000], ['r', 'b', 'g', 'm', 'y', 'k']):
    
    eigs, _ = np.linalg.eig(Ws[i,:,:])
    #plt.figure()
    plt.plot(np.real(eigs), np.imag(eigs), '.', color=col)

theta = np.arange(0, 2*np.pi, 0.01)
plt.plot(np.cos(theta), np.sin(theta))
plt.xlim([-1.5, 1.5])
plt.ylim([-1.5, 1.5])

def get_spectral_radius(M):
    
    eigs, _ = np.linalg.eig(M)
    
    return np.amax(np.absolute(eigs))

r = []
for i in range(Ws.shape[0]):
    
    r.append(get_spectral_radius(As[i,:,:]))
    
smoothed_loss = np.convolve(losses, np.ones(100)/100, mode='valid')
smoothed_radii = np.convolve(r, np.ones(100)/100, mode='valid')
plt.plot([0, len(smoothed_loss)], [0.66, 0.66], '--', color='r')
plt.plot([0, len(smoothed_loss)], [0.52, 0.52], '--', color='m')
plt.plot([0, len(smoothed_loss)], [0.45, 0.45], '--', color='g')
plt.ylim([0,2])
plt.xlim([1000, 4000])
plt.plot(smoothed_loss)
plt.plot(smoothed_radii)























