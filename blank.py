#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 14:23:44 2018

@author: omarschall
"""

#rnn.run(data,
#        learn_alg=learn_alg,
#        optimizer=optimizer,
#        monitors=monitors,
#        update_interval=1,
#        l2_reg=0.01,
#        check_accuracy=False,
#        verbose=False,
#        t_stop_training=50)

rnn.run(data, monitors=['loss_', 'a'], mode='test')
a = rnn.mons['a']
U, S, V = np.linalg.svd(a)
x = data['test']['X'][:,0]
traj = a.dot(V)
PC1 = traj[:,0]

n_pc = 4
n_roll = 14

def d_prime(A, B):
    
    return (np.mean(A) - np.mean(B))/np.sqrt(0.5*(np.var(A) + np.var(B)))

for i_pc in range(n_pc):
    PC = traj[:,i_pc]
    d_primes = []
    for i_roll in range(n_roll):
    
        A = PC[np.where(np.roll(x,i_roll)==0)]
        B = PC[np.where(np.roll(x,i_roll)==1)]
        
        d_primes.append(d_prime(A,B))
        
    plt.plot(d_primes)
    
plt.legend(['PC{}'.format(i) for i in range(n_pc)])
plt.xticks(range(n_roll))
plt.xlabel('Time Lag')
plt.ylabel("d'")
plt.axhline(y=0, color='k', linestyle='--')