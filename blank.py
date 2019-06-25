#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 14:23:44 2018

@author: omarschall
"""

import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
from utils import *


x = [0, 1]
for i in range(1, 6):
    plt.plot(x, [alignment_means[0, i], alignment_means[-1, i]], color='C0')
    plt.plot(x, [alignment_means[0, i], alignment_means[-1, i]], '.', color='C0')
plt.plot(x, [alignment_means[0, 6], alignment_means[-1, 6]], color='C1')
plt.plot(x, [alignment_means[0, 6], alignment_means[-1, 6]], '.', color='C1')

A = np.random.normal(0, 1, (3000, 3000))
B = np.random.normal(0, 1, (3000, 3000))
x = np.random.normal(0, 1, 3000)

t1 = time.time()

#y = A.dot(B.dot(x))
y = (A.dot(B)).dot(x)

t2 = time.time()

print(t2 - t1)

#if params['alpha'] == 1:
#    n_1, n_2 = 6, 10
#    tau_task = 1
#    optimizer = SGD(lr=0.0001)
#if params['alpha'] == 0.5:
#    n_1, n_2 = 4, 7
#    tau_task = 2
#    optimizer = SGD(lr=0.001)

#task = Coin_Task(4, 7, one_hot=True, deterministic=True, tau_task=3)

#np.random.seed(10)

#n_in = 32
#n_hidden = 32
#n_out = 32
#
#W_in_target  = np.random.normal(0, np.sqrt(1/(n_in)), (n_hidden, n_in))
#W_rec_target = np.linalg.qr(np.random.normal(0, 1, (n_hidden, n_hidden)))[0]
##W_rec_target = np.random.normal(0, np.sqrt(1/n_hidden), (n_hidden, n_hidden))
#W_out_target = np.random.normal(0, np.sqrt(1/(n_hidden)), (n_out, n_hidden))
#b_rec_target = np.random.normal(0, 0.1, n_hidden)
#b_out_target = np.random.normal(0, 0.1, n_out)
#
#alpha = 1
#
#rnn_target = RNN(W_in_target, W_rec_target, W_out_target,
#                 b_rec_target, b_out_target,
#                 activation=tanh,
#                 alpha=alpha,
#                 output=identity,
#                 loss=mean_squared_error)
#
#task = Mimic_RNN(rnn_target, p_input=0.5, tau_task=1)

### --- Define Layer Normalization --- ###

def layer_normalization_(z):
    
    return (z - np.mean(z))/np.std(z)

def layer_normalization_derivative(z):
    
    return "don't care"

layer_normalization = function(layer_normalization_, layer_normalization_derivative)
            
if hasattr(self, 't_stop_SG_train'):
    if self.t_stop_SG_train==i_t:
        self.learn_alg.optimizer.lr = 0

def forward_estimate_credit_assignment(self, i_t, data, t_steps=14, delta_a=0.0001):
    
    try:
        truncated_data = {'test': {'X': data['train']['X'][i_t:i_t+t_steps,:],
                                   'Y': data['train']['Y'][i_t:i_t+t_steps,:]}}
    except IndexError:
        return
    
    fiducial_rnn = copy(self.net)
    fiducial_sim = copy(self)
    perturbed_rnn = copy(self.net)
    perturbed_sim = copy(self)
    
    direction = np.random.normal(0, 1, self.net.n_hidden)
    perturbation = delta_a*direction/norm(direction)
    a_fiducial = self.net.a - perturbation
    a_perturbed = self.net.a + perturbation
    
    fiducial_sim.run(truncated_data,
                     mode='test',
                     monitors=['loss_'],
                     a_initial=a_fiducial,
                     verbose=False)
    
    perturbed_sim.run(truncated_data,
                      mode='test',
                      monitors=['loss_'],
                      a_initial=a_perturbed,
                      verbose=False)
    
    delta_loss = perturbed_sim.mons['loss_'].sum() - fiducial_sim.mons['loss_'].sum()
    self.CA_forward_est = delta_loss/(2*delta_a)
    self.CA_SG_est = self.learn_alg.sg.dot(direction)

if False:
    data_dir = '/scratch/oem214/vanilla-rtrl/library/ssa_2_run'
    
    i_file = i_job
    file_name = 'rnn_{}'.format(i_file)
    test_file_name = 'rnn_{}_test_data'.format(i_file)
    
    data_path = os.path.join(data_dir, test_file_name)
    rnn_path = os.path.join(data_dir, file_name)
    with open(data_path, 'rb') as f:
        test_data = pickle.load(f)
    with open(rnn_path, 'rb') as f:
        result = pickle.load(f)
        
    n_trials = len(test_data.keys())
    alignments = np.zeros((n_trials, n_trials))
    for i in range(n_trials):
        for j in range(n_trials):
            alignments[i,j] = np.square(test_data[i]['PCs'].T.dot(test_data[j]['PCs'])).sum()/3
    
    result = {}
    result[file_name+'_alignments'] = alignments
    
    if os.environ['HOME']=='/home/oem214':
    
        save_dir = os.environ['SAVEPATH']
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        save_path = os.path.join(save_dir, 'rnn_{}_analysis'.format(i_file))
        
        with open(save_path, 'wb') as f:
            pickle.dump(result, f)

data_dir = '/scratch/oem214/vanilla-rtrl/library/ssa_learning_rtrl_sg'

i_file = i_job
file_name = 'rnn_{}'.format(i_file)
test_file_name = 'rnn_{}_test_data'.format(i_file)

data_path = os.path.join(data_dir, test_file_name)
rnn_path = os.path.join(data_dir, file_name)
with open(data_path, 'rb') as f:
    test_data = pickle.load(f)
with open(rnn_path, 'rb') as f:
    result = pickle.load(f)
    
n_trials = len(test_data.keys())
alignments = np.zeros((n_trials, n_trials))
for i in range(n_trials):
    for j in range(n_trials):
        alignments[i,j] = np.square(test_data[i]['PCs'].T.dot(test_data[j]['PCs'])).sum()/3

result = {}
result[file_name+'_alignments'] = alignments

if os.environ['HOME']=='/home/oem214':

    save_dir = os.environ['SAVEPATH']
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_path = os.path.join(save_dir, 'rnn_{}_analysis'.format(i_file))
    
    with open(save_path, 'wb') as f:
        pickle.dump(result, f)

##-----

job_name = 'ssa_learning_bptt_sg'
data_dir = os.path.join('/Users/omarschall/cluster_results/vanilla-rtrl/', job_name)

data_dir = '/Users/omarschall/vanilla-rtrl/library'
file_name = 'rnn_0_trial_0'
with open(os.path.join(data_dir, file_name), 'rb') as f:
    #result = pickle.load(f)
    test_data = pickle.load(f)

for i in range(test_data['PC_on_trajs'].shape[0]):
    plt.plot(test_data['PC_on_trajs'][i,:,0],test_data['PC_on_trajs'][i,:,1], color='b', alpha=0.2)
for i in range(test_data['PC_off_trajs'].shape[0]):
    plt.plot(test_data['PC_off_trajs'][i,:,0],test_data['PC_off_trajs'][i,:,1], color='g', alpha=0.2)
    
if hasattr(self, 'time_steps_per_trial'):
    X_reshaped = data['test']['X'].reshape((-1, self.time_steps_per_trial, self.net.n_in))
    on_trials = np.where(X_reshaped[:,1,0]>0)[0]
    off_trials = np.where(X_reshaped[:,1,0]<0)[0]
    
#Initialize test data
if hasattr(self, 'SSA_PCs') and mode=='train':
    test_data = {}
    file_name = 'rnn_{}_test_data'.format(self.i_job)
    save_path = os.path.join(self.save_dir, file_name)
    with open(save_path, 'wb') as f:
        pickle.dump(test_data, f)
        
if hasattr(self, 'SSA_PCs') and self.mode=='train':
    
    with open(save_path, 'rb') as f:
        test_data = pickle.load(f)
    
    np.random.seed(0)
    test_sim = copy(self)
    test_sim.run(data, mode='test', monitors=['a'], verbose=False)
    
    test_data_trial = {}
    PCs = State_Space_Analysis(test_sim.mons['a'], add_fig=False).V[:,:self.SSA_PCs]
    A = test_sim.mons['a'].reshape((-1, self.time_steps_per_trial, self.net.n_hidden))
    PC_on_trajs = A[on_trials].dot(PCs)
    PC_off_trajs = A[off_trials].dot(PCs)
    test_data_trial['PCs'] = PCs
    test_data_trial['PC_on_trajs'] = PC_on_trajs
    test_data_trial['PC_off_trajs'] = PC_off_trajs
    
    test_data[self.i_trial] = test_data_trial
    with open(save_path, 'wb') as f:
        pickle.dump(test_data, f)
#sim = result['sim']
#print(result['config'])
#
#n_trials = len(sim.SSAs)
#n_PCs = 3
#alignments = np.zeros((n_trials, n_trials))
#for i in range(n_trials):
#    for j in range(n_trials):
#        alignments[i,j] = (norm(sim.SSAs[i][:,:n_PCs].T.dot(sim.SSAs[j][:,:n_PCs]))**2)/n_PCs
#        
#plt.imshow(alignments)


#a = rnn.mons['a']
#T = 100
#n_tau = 5
#Y = []
#for t in range(a.shape[0]//T):
#    a_t = a[t*T:(t+1)*T]
#    autocov = np.zeros((a.shape[1], n_tau))
#    for i in range(a_t.shape[1]):
#        for tau in range(1, 1+n_tau):
#        
#            autocov[i, tau-1] = np.corrcoef(a_t[:,i], np.roll(a_t[:,i], tau, axis=0))[0,1]
#        
#        #plt.plot(autocov[i, :], 'b', alpha=0.2)
#    Y.append(autocov.mean())
#
##plt.plot(autocov.mean(0), 'b')
#plt.plot(Y)
#plt.ylim([-1, 1])
#plt.ylabel('Autoorrelation')
#plt.xlabel('Time')
#plt.xticks(range(0, n_tau, 4))

#signals = []
#legend = []
#for key in sim.mons.keys():
#    s = sim.mons[key].shape
#    if len(s)==1 and s[0]>0:
#        signals.append(sim.mons[key])
#        legend.append(key)
#fig1 = plot_filtered_signals(signals, filter_size=100, y_lim=[0, 1])
#plt.legend(legend)
#fig2 = plot_filtered_signals(signals, filter_size=100, y_lim=[0, 20])
#plt.legend(legend)
#
#try:
#    plt.figure()
#    dots = (sim.mons['sg'][:-(T-1)]*sim.mons['CA']).sum(1)
#    norms = np.sqrt(np.square(sim.mons['sg'][:-(T-1)]).sum(1)*np.square(sim.mons['CA']).sum(1))
#    plt.plot(dots/norms, '.', alpha=0.3)
#except:
#    pass

### -------- Make big fig ---- ####

job_name = 'alpha03_tau4_checkpoints'
data_dir = os.path.join('/Users/omarschall/cluster_results/vanilla-rtrl/', job_name)
#alpha = 0.05
#filter_size = 100

configs = []
signals = []
figs = []

n_errors = 0

n_row = 3
n_col = 3

loss_avg_1 = [0]*30
loss_avg_2 = [0]*30
loss_avg_3 = [0]*30

n_files = len(os.listdir(data_dir)) - 1

fig, axarr = plt.subplots(n_row, n_col, figsize=(20, 10))

val_losses = {}

#for file_name in os.listdir(data_dir):
for i_file in [29]:
    
    file_name = 'rnn_'+str(i_file)
    
    if 'code' in file_name or '.' in file_name:
        continue
    
    file_no = int(file_name.split('_')[-1])
    
    with open(os.path.join(data_dir, file_name), 'rb') as f:
        try:
            result = pickle.load(f)
        except EOFError:
            n_errors += 1
    
    #plt.figure(figsize=(10, 20))
    for i in range(9):
        i_x = i%n_row
        i_y = i//n_row
        np.random.seed(i+100)
        data = result['task'].gen_data(1000, 5000)
        

    
        test_sim = Simulation(result['sim'].best_net, learn_alg=None, optimizer=None)
        test_sim.run(data, mode='test', monitors=['loss_', 'y_hat'], verbose=False)
        axarr[i_x, i_y].plot(test_sim.mons['y_hat'][:,0])
        axarr[i_x, i_y].plot(data['test']['Y'][:,0], alpha=0.4)
        axarr[i_x, i_y].set_xlim([1000, 1400])
        #axarr[i_x, i_y].plot(test_sim.mons['y_hat'][:,0], data['test']['Y'][:,0], '.', alpha=0.01)
        #axarr[i_x, i_y].plot([0, 1], [0, 1], 'k', linestyle='--')
        title = 'Seed = {}'.format(i)
        #axarr[i_x, i_y].set_title(title)
        axarr[i_x, i_y].set_xticks([])
        axarr[i_x, i_y].set_yticks([])
    
    continue 
#    config = [result['sim'].net.alpha, result['task'].tau_task]
#    
#    if config not in configs:
#        configs.append(config)
#        i_conf = len(configs) - 1
#        first = True
#    else:
#        i_conf = configs.index(config)
#        first = False
#    
#    if config==[0.3, 4]:
#        i_seed = result['i_seed']
#        if i_seed==17:
#            break
#    else:
#        continue
    
    np.random.seed(100)
    data = result['task'].gen_data(1000, 5000)
        
    i_seed = result['i_seed']
    i_x = i_seed%n_row
    i_y = i_seed//n_row
    
    test_sim = Simulation(result['sim'].best_net, learn_alg=None, optimizer=None)
    test_sim.run(data, mode='test', monitors=['loss_', 'y_hat'], verbose=False)
    axarr[i_x, i_y].plot(test_sim.mons['y_hat'][:,0])
    axarr[i_x, i_y].plot(data['test']['Y'][:,0], alpha=0.4)
    axarr[i_x, i_y].set_xlim([1000, 1100])
    #axarr[i_x, i_y].plot(test_sim.mons['y_hat'][:,0], data['test']['Y'][:,0], '.', alpha=0.01)
    #axarr[i_x, i_y].plot([0, 1], [0, 1], 'k', linestyle='--')
    title = 'Seed = {}'.format(i_seed)
    #axarr[i_x, i_y].set_title(title)
    axarr[i_x, i_y].set_xticks([])
    axarr[i_x, i_y].set_yticks([])
        
    val_losses[file_name] = result['sim'].best_val_loss
    
    continue
        
    i_x = i_conf%n_row
    i_y = i_conf//n_row
    
    #np.random.seed(result['i_seed'])
    #data = result['task'].gen_data(1000, 10000)
    
    #sim = Simulation(result['sim'].net, learn_alg=None, optimizer=None)
    #sim.run(data, mode='test', monitors=['loss_', 'y_hat'], verbose=False)
    #axarr[i_x, i_y].plot(sim.mons['y_hat'][:,0])
    #axarr[i_x, i_y].plot(data['test']['Y'][:,0], alpha=0.4)
    
    loss_1 = rectangular_filter(result['sim'].mons['loss_'], filter_size=1000)
    #loss_2 = rectangular_filter(result['sim'].mons['sg_loss'], filter_size=1000)
    #loss_3 = rectangular_filter(result['sim'].mons['loss_a'], filter_size=1000)
    
    loss_avg_1[i_conf] += loss_1
    #loss_avg_2[i_conf] += loss_2
    #loss_avg_3[i_conf] += loss_3
    
    axarr[i_x, i_y].plot(loss_1, color='b', alpha=0.05)
    #axarr[i_x, i_y].plot(loss_2, color='y', alpha=0.05)
    #axarr[i_x, i_y].plot(loss_3, color='r', alpha=0.05)
    
    if first:
        axarr[i_x, i_y].axhline(y=0.66, color='r', linestyle='--')
        axarr[i_x, i_y].axhline(y=0.52, color='m', linestyle='--')
        axarr[i_x, i_y].axhline(y=0.45, color='g', linestyle='--')
    #axarr[i_x, i_y].plot(result['sim'].mons['loss_'])
    #axarr[i_x, i_y].plot(result['sim'].mons['y_hat'][:,0])
    #axarr[i_x, i_y].plot(data['train']['Y'][:,0])
        title = 'Alpha = {}, Tau = {}'.format(config[0], config[1])
        axarr[i_x, i_y].set_title(title)
        axarr[i_x, i_y].set_xticks([])
        axarr[i_x, i_y].set_yticks([])
        #axarr[i_x, i_y].set_xlim([1000, 4000])
        axarr[i_x, i_y].set_ylim([0, 0.8])

n_seeds = 20
for i_conf in range(len(configs)):
    
    i_x = i_conf%n_row
    i_y = i_conf//n_row
    
    axarr[i_x, i_y].plot(loss_avg_1[i_conf]/n_seeds, color='b')
    #axarr[i_x, i_y].plot(loss_avg_2[i_conf]/n_seeds, color='y')
    #axarr[i_x, i_y].plot(loss_avg_3[i_conf]/n_seeds, color='r')    

### STATE SPACE STUFF ###
#State space
#plt.figure()
#ssa = State_Space_Analysis(test_sim.mons['a'], n_PCs=3)
#for i, col in enumerate(['C{}'.format(i_col) for i_col in range(8)]):
#    cond = np.array([True]*(n_test-6))
#    prev_inputs = [int(s) for s in bin(i)[2:].zfill(5)]
#    for i_back, prev_input in enumerate(prev_inputs):
#        set_trace()
#        cond_ = data['test']['X'][5-i_back:-1-i_back,0]==prev_input
#        cond = np.logical_and(cond, cond_)
#    print(cond.sum())
#    ssa.plot_in_state_space(test_sim.mons['a'][5:-1][cond], '.', alpha=0.1, color=col)
#    
##    for past in [[0,0], [0, 1], [1, 0], [1, 1]]:
##        cond = np.where(np.logical_and(data['test']['X'][2:-2,0]==past[0], data['test']['X'][4:,0]==past[1]))
##        ssa.plot_in_state_space(test_sim.mons['a'][:-4,][cond], '.', alpha=0.3)
#    
#for y in [0.25, 0.5, 0.75, 1]:
#    cond = np.where(data['test']['Y'][:,0]==y)
#    ssa.plot_in_state_space(test_sim.mons['a'][cond], '.', alpha=0.3)
    
#ssa.plot_in_state_space(test_sim.mons['a'])


#    for key, col in zip(['loss_', 'acc'], ['b', 'k']):
#        smoothed_signal = np.convolve(result['rnn'].mons[key],
#                                      np.ones(filter_size)/filter_size,
#                                      mode='valid')
#        axarr[i_x, i_y].plot(smoothed_signal, col, alpha=alpha)
#        signals[i_conf][key].append(smoothed_signal)
#        
#    for key, col in zip(['sg_loss', 'loss_a'], ['y', 'g']):
#        
#        smoothed_signal = np.convolve(result['rnn'].learn_alg.mons[key],
#                                      np.ones(filter_size)/filter_size,
#                                      mode='valid')
#        axarr[i_x, i_y].plot(smoothed_signal, col, alpha=alpha)
#        signals[i_conf][key].append(smoothed_signal)
#
#for i_conf, conf in enumerate(configs):
#
#    i_x = i_conf%n_row
#    i_y = i_conf//n_row
#    
#    for key, col in zip(['loss_', 'acc', 'sg_loss', 'loss_a'], ['b', 'k', 'y', 'g']):
#        
#        signals[i_conf][key] = np.array(signals[i_conf][key])
#        
#        axarr[i_x, i_y].plot(np.nanmean(signals[i_conf][key], axis=0), col)
#    
#    if True:
#        axarr[i_x, i_y].axhline(y=0.66, color='r', linestyle='--')
#        axarr[i_x, i_y].axhline(y=0.52, color='m', linestyle='--')
#        axarr[i_x, i_y].axhline(y=0.45, color='g', linestyle='--')    
#        axarr[i_x, i_y].axhline(y=0.75, color='k', linestyle='--')   
#        
#    axarr[i_x, i_y].set_ylim([0, 1])
#    axarr[i_x, i_y].set_xlim([0, 10000])
#    axarr[i_x, i_y].set_xticks([])
#    axarr[i_x, i_y].set_title('{}, {}, {}'.format(conf[0], conf[1], conf[2]), fontsize=8)
#    
#print(n_errors)
#### ------------- #####

#import os
#
#figs_path = '/Users/omarschall/weekly-reports/report_10-31-2018/figs'
#
#
#fig = plt.figure(figsize=(5,5))
#plt.plot(rnn.mons['y_hat'][10000:,0], data['train']['Y'][10000:,0], '.', alpha=0.005)
#plt.xlim([0, 1])
#plt.ylim([0, 1])
#plt.axis('equal')
#plt.xticks([0, 1])
#plt.yticks([0, 1])
#x_ = np.linspace(0, 1, 10)
#plt.plot(x_, x_, color='k', linestyle='--')
#plt.xlabel('Predicted')
#plt.ylabel('Label')
##fig.savefig(os.path.join(figs_path, 'Fig1.png'), dpi=200, format='png')
#
#
#signals = [rnn.mons['loss_'], rnn.learn_alg.mons['sg_loss']]
#fig = plot_filtered_signals(signals, filter_size=100, y_lim=[0, 1.5])
#plt.xlabel('Time')
#plt.legend(['Loss', 'SG Loss'])
#fig.savefig(os.path.join(figs_path, 'Fig2.png'), dpi=200, format='png')

#from analysis_funcs import get_spectral_radius
#
#avg_eval_mod = []
#spectral_radii = []
#for W_rec in rnn.mons['W_rec']:
#    
#    eigs, vecs = np.linalg.eig(W_rec)
#    avg_eval_mod.append(np.absolute(eigs).mean())
#    spectral_radii.append(np.amax(np.absolute(eigs)))
#    
#plt.plot(avg_eval_mod)
#plt.plot(spectral_radii)
#plt.legend(['Avg eigenvalue modulus', 'Spectral radius'])


#rnn.run(data,
#        learn_alg=learn_alg,
#        optimizer=optimizer,
#        monitors=monitors,
#        update_interval=1,
#        l2_reg=0.01,
#        check_accuracy=False,
#        verbose=False,
#        t_stop_training=50)

#rnn.run(data, monitors=['loss_', 'a'], mode='test')
#a = rnn.mons['a']
#U, S, V = np.linalg.svd(a)
#x = data['test']['X'][:,0]
#traj = a.dot(V)
#PC1 = traj[:,0]
#
#n_pc = 4
#n_roll = 14
#
#def d_prime(A, B):
#    
#    return (np.mean(A) - np.mean(B))/np.sqrt(0.5*(np.var(A) + np.var(B)))
#
#for i_pc in range(n_pc):
#    PC = traj[:,i_pc]
#    d_primes = []
#    for i_roll in range(n_roll):
#    
#        A = PC[np.where(np.roll(x,i_roll)==0)]
#        B = PC[np.where(np.roll(x,i_roll)==1)]
#        
#        d_primes.append(d_prime(A,B))
#        
#    plt.plot(d_primes)
#    
#plt.legend(['PC{}'.format(i) for i in range(n_pc)])
#plt.xticks(range(n_roll))
#plt.xlabel('Time Lag')
#plt.ylabel("d'")
#plt.axhline(y=0, color='k', linestyle='--')

#n_train = 100000
#n_test  = 1000
#
#text = open('/Users/omarschall/datasets/shakespeare.txt', 'r').read() # should be simple plain text file
#chars = list(set(text))
#data_size, vocab_size = len(text), len(chars)
#print('data has {} characters, {} unique.'.format(data_size, vocab_size))
#char_to_ix = { ch:i for i,ch in enumerate(chars) }
#ix_to_char = { i:ch for i,ch in enumerate(chars) }
#
#input_data = np.zeros((n_train, 84))
#output_data = np.zeros((n_train, 84))
#for i in range(1, n_train):
#    
#    k = char_to_ix[text[i]]
#    input_data[i,k] = 1
#    output_data[i-1, k] = 1
#    
#data = {}
#data['train'] = {'X': input_data, 'Y': output_data}
#
#input_data = np.zeros((n_test, 84))
#output_data = np.zeros((n_test, 84))
#
#for i in range(n_train, n_train+n_test):
#    
#    k = char_to_ix[text[i]]
#    input_data[i-n_train,k] = 1
#    output_data[i-1-n_train, k] = 1
#
#data['test'] = {'X': input_data, 'Y': output_data}
#
#data['test'] = {'X': np.zeros((1000, 84)), 'Y': np.zeros((1000, 84))}
#
#rnn.run(data, mode='test', monitors=['y_hat'])