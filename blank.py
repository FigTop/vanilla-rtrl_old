#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 14:23:44 2018

@author: omarschall
"""


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

### -------- Make big fig ---- ####

job_name = 'approximations'
data_dir = os.path.join('/Users/omarschall/cluster_results/vanilla-rtrl/', job_name)
alpha = 0.05
filter_size = 100

configs = []
signals = []
figs = []

fig, axarr = plt.subplots(2, 2, figsize=(20, 10))

for file_name in os.listdir(data_dir):
    
    with open(os.path.join(data_dir, file_name), 'rb') as f:
        result = pickle.load(f)
        
    if result['config'] not in configs:
        configs.append(result['config'])
        signals.append({'loss_': [], 'sg_loss': [], 'loss_a': []})
        i_conf = len(configs) - 1
    else:
        i_conf = configs.index(result['config'])
    
    i_x = i_conf%2
    i_y = i_conf//2
    
    smoothed_signal = np.convolve(result['rnn'].mons['loss_'],
                                  np.ones(filter_size)/filter_size,
                                  mode='valid')
    axarr[i_x, i_y].plot(smoothed_signal, 'b', alpha=alpha)
    signals[i_conf]['loss_'].append(smoothed_signal)
        
    for key, col in zip(['sg_loss', 'loss_a'], ['y', 'g']):
        
        smoothed_signal = np.convolve(result['rnn'].learn_alg.mons[key],
                                      np.ones(filter_size)/filter_size,
                                      mode='valid')
        axarr[i_x, i_y].plot(smoothed_signal, col, alpha=alpha)
        signals[i_conf][key].append(smoothed_signal)

for i_conf, conf in enumerate(configs):

    i_x = i_conf%2
    i_y = i_conf//2
    
    for key, col in zip(['loss_', 'sg_loss', 'loss_a'], ['b', 'y', 'g']):
        
        signals[i_conf][key] = np.array(signals[i_conf][key])
        
        axarr[i_x, i_y].plot(np.nanmedian(signals[i_conf][key], axis=0), col)
    
    if True:
        axarr[i_x, i_y].axhline(y=0.66, color='r', linestyle='--')
        axarr[i_x, i_y].axhline(y=0.52, color='m', linestyle='--')
        axarr[i_x, i_y].axhline(y=0.45, color='g', linestyle='--')    
        
    axarr[i_x, i_y].set_ylim([0, 1.5])
    axarr[i_x, i_y].set_xticks([])
    axarr[i_x, i_y].set_title('{}, {}, {}, {}'.format(conf[0], conf[1], conf[2], conf[3]), fontsize=8)
    

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