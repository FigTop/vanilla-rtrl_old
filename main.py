#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 16:30:58 2018

@author: omarschall
"""

import numpy as np
from network import RNN
from utils import *
from gen_data import *
import matplotlib.pyplot as plt
import time
from optimizers import *
from analysis_funcs import *
from learning_algorithms import *

np.random.seed(1)

task = Coin_Task(3, 6, one_hot=True, deterministic=False)
data = task.gen_data(15000, 1000)
#task = Copy_Task(10, 3)

n_in     = task.n_in
n_hidden = 32
n_out    = task.n_out

W_in  = np.random.normal(0, np.sqrt(1/(n_in)), (n_hidden, n_in))
W_rec = np.random.normal(0, np.sqrt(1/(n_hidden)), (n_hidden, n_hidden))
W_out = np.random.normal(0, np.sqrt(1/(n_hidden)), (n_out, n_hidden))

b_rec = np.zeros(n_hidden)
b_out = np.zeros(n_out)

alpha = 1

rnn = RNN(W_in, W_rec, W_out, b_rec, b_out,
          activation=relu,
          alpha=alpha,
          output=softmax,
          loss=softmax_cross_entropy)


optimizer = SGD(lr=0.001)
SG_optimizer = SGD(lr=0.01)
learn_alg = DNI(rnn, SG_optimizer, monitors=['sg_loss', 'sg', 'A', 'B', 'C', 'SG_grads'],
                lambda_mix=0, l2_reg=0, fix_SG_interval=1)
comp_alg = KF_RTRL(rnn, monitors=['A'])
monitors = ['loss_', 'alignment', 'y_hat', 'a', 'y', 'W_rec', 'grads']

rnn.run(data,
        learn_alg=learn_alg,
        optimizer=optimizer,
        monitors=monitors,
        update_interval=1,
        l2_reg=0.001,
        check_accuracy=True,
        comparison_alg=comp_alg)

plt.figure()
leg = []
for obj in [rnn, learn_alg]:
    mons = getattr(obj, 'mons')
    for key in mons.keys():
        signal = []
        if 'grads' not in key:
            for i in range(len(mons[key])):
                x = mons[key][i]
                signal.append(np.mean(x**2))
        else:
            for i in range(len(mons[key])):
                g = mons[key][i]
                signal.append(np.concatenate([np.square(g_.flatten()) for g_ in g]).mean())
                
        plt.plot(signal)
        leg.append(key)
        
        
plt.xlim([3100, 3124])
plt.ylim([0, 30])
plt.legend(leg)

#W = np.nan_to_num(np.array(rnn.mons['W_rec']))
#A = np.nan_to_num(np.array(learn_alg.mons['A']))
#W_radii = get_spectral_radii(W)
#A_radii = get_spectral_radii(A)
#
#plt.figure()
#task.plot_filtered_signals([W_radii, A_radii], plot_loss_benchmarks=False)
#
#fig = plt.figure()
#signals = [rnn.mons['loss_'], learn_alg.mons['sg_loss']]
#task.plot_filtered_signals(signals, y_lim=[0, 1.5])
#plt.legend(['Loss', 'SG Loss'])
#
#if True:
#    fig = plt.figure()
#    signals = [[a[i] for a in rnn.mons['alignment']] for i in range(3)]
#    task.plot_filtered_signals(signals, y_lim=[-1.2, 1.2], plot_loss_benchmarks=False)
#    plt.legend(['W_rec', 'W_in', 'b_rec'])
#    plt.ylabel('Normalized Dot Product')


#A_DNI = np.array(learn_alg.mons['A'])
#dAdt_DNI = A_DNI[1:] - A_DNI[:-1]
#A_KF = np.array(comp_alg.mons['A'])
#dAdt_KF = A_KF[1:] - A_KF[:-1]
#
#alignment = get_vector_alignment(dAdt_DNI, dAdt_KF)
#
#
#norm_DNI = (dAdt_DNI**2).mean(1).mean(1)
#norm_KF = (dAdt_KF**2).mean(1).mean(1)






























