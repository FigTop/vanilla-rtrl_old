#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 23:02:35 2018

@author: omarschall
"""

import numpy as np
from network import RNN
from utils import *
#from gen_data import gen_data
import matplotlib.pyplot as plt
import time
from optimizers import *
import pickle
import os
from analysis_funcs import *

figs_path = '/Users/omarschall/weekly-reports/report_10-25-2018/figs'
fig = plt.figure(figsize=(8,4))
colors = ['b', 'y']#, 'g', 'r', 'm']
for color in colors:
    plt.plot([], color)
plt.legend(['Loss', 'SG Loss'])
#plt.legend(['W_rec', 'W_in', 'b_rec', 'W_out', 'b_out'])
plot_results_from_job('dni_compare_bptt',
                      rnn_signals=['loss_'],#p+'_alignment' for p in ['W_rec', 'W_in', 'b_rec', 'W_out', 'b_out']],
                      learn_alg_signals=['sg_loss'],
                      colors=colors,
                      alpha=0.01,
                      n_seeds=20,
                      y_lim=[0, 1.5],
                      plot_loss_benchmarks=True)

plt.title('Synthetic Gradients')
#plt.ylabel('Normalized Dot Product')
#plt.title('Backpropagation Through Time (10-step truncation)')
plt.axhline(y=0, color='k', linestyle='--')

fig.savefig(os.path.join(figs_path, 'Fig1.png'), dpi=200, format='png')


















