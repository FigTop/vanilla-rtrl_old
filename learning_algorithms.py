#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 15:03:11 2018

@author: omarschall
"""

class BPTT:
    
    def __init__(self, network, t1, t2):
        
        #RNN instance that BPTT is being applied to
        self.network = network
        
        #The two integer parameters
        self.t1  = t1    #Number of steps to average loss over
        self.t2  = t2    #Number of steps to propagate backwards
        self.T   = n1+n2 #Total amount of network hist needed to memorize
        
        #Lists storing relevant network hist
        self.h_hist = [np.zeros(self.network.n_hidden) for _ in range(self.T)]
        self.a_hist = [np.zeros(self.network.n_hidden) for _ in range(self.T)]
        self.e_hist = [np.zeros(self.network.n_hidden) for _ in range(t1)]
        self.x_hist = [np.zeros(self.network.n_hidden) for _ in range(self.T)]
        
    def update_learning_vars(self):
        '''
        Must be run every time step
        '''
        
        for attr in ['h', 'a', 'e', 'x']:
            self.__dict__[attr+'_hist'].append(getattr(self.network, attr))
            del(self.__dict__[attr+'_hist'][0])
    
    def __call__(self):
        '''
        Run only when grads are needed for the optimizer
        '''
        
        W_out_grad = [np.mulitply.outer(self.e_hist[-i],self.a_hist[-i]) for i in range(1, self.t1+1)]
        W_out_grad = sum(W_out_grad)/self.t1
        
        b_out_grad = [self.e_hist[-i] for i in range(1, self.t1+1)]
        b_out_grad = sum(b_out_grads)/self.t1
        
        outer_grads = [W_out_grad, b_out_grad]
        
        get_a_J = self.network.get_a_jacobian
        
        a_Js = [get_a_J(update=False, h=self.h_hist[-i], h_prev=self.h_hist[-(i+1)]) for i in range(1, self.T)]
        
        n_hidden, n_in = self.network.n_hidden, self.network.n_in
        rec_grad = np.zeros((n_hidden, n_hidden+n_in+1))
        
        for i in range(1, self.t1+1):
            
            q = self.e_hist[-i].dot(self.network.W_out)
            
            for j in range(self.t2):
                
                J = np.eye(n_hidden)
                
                for k in range(j):
                    
                    J = J.dot(a_Js[-(i+k)])
                
                J = J.dot(np.diag(self.network.activation.f_prime(self.h_hist[-(i+j)])))
                
                pre_activity = np.concatenate([self.h_hist[-(i+j+1)], self.x_hist[-(i+j+1)], np.array([1])])
                rec_grad += np.multiply.outer(q.dot(J), pre_activity)
        
        grads = [rec_grads[:,:n_hidden], rec_grad[:,n_hidden:-1], rec_grad[:,-1]] + outer_grads
        
        return grads
        