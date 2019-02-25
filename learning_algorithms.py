#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 15:03:11 2018

@author: omarschall
"""

import numpy as np
from pdb import set_trace
from utils import *
from functions import *
from copy import copy

class Learning_Algorithm:
    
    def __init__(self, net, allowed_kwargs_, **kwargs):
        
        allowed_kwargs = {'W_FB'}.union(allowed_kwargs_)
                
        for k in kwargs:
            if k not in allowed_kwargs:
                raise TypeError('Unexpected keyword argument '
                                'passed to Learning_Algorithm.__init__: ' + str(k))
        
        self.__dict__.update(kwargs)
        
        #RNN instance the algorithm is being applied to
        self.net = net
        self.n_in  = self.net.n_in
        self.n_h   = self.net.n_hidden
        self.n_out = self.net.n_out
        self.q     = np.zeros(self.n_h)
        
class Real_Time_Learning_Algorithm(Learning_Algorithm):
    
    def get_outer_grads(self):

        return [np.multiply.outer(self.net.e, self.net.a), self.net.e]
    
    def propagate_feedback_to_hidden(self):
        
        self.q_prev = np.copy(self.q)
        
        if not hasattr(self, 'W_FB'):
            self.q = self.net.e.dot(self.net.W_out)
        else:
            self.q = self.net.e.dot(self.W_FB)
    
    def __call__(self):
        
        outer_grads = self.get_outer_grads()
        self.propagate_feedback_to_hidden()
        rec_grads = split_weight_matrix(self.get_rec_grads(), [self.n_h, self.n_in, 1])
        grads = rec_grads + outer_grads
        
        return grads
        
class RTRL(Real_Time_Learning_Algorithm):

    def __init__(self, net, **kwargs):
        
        allowed_kwargs_ = set()
        super().__init__(net, allowed_kwargs_, **kwargs)
        
        #Initialize influence matrix
        self.dadw  = np.zeros((self.n_h, self.net.n_hidden_params))
        
    def update_learning_vars(self):
        
        #Get relevant values and derivatives from network
        self.a_hat = np.concatenate([self.net.a_prev, self.net.x, np.array([1])])
        self.papw = np.kron(self.a_hat, np.diag(self.net.activation.f_prime(self.net.h)))
        self.net.get_a_jacobian()
        
        #Update influence matrix
        self.dadw = self.net.a_J.dot(self.dadw) + self.papw
        
    def get_rec_grads(self):

        return self.q.dot(self.dadw).reshape((self.n_h, self.n_h + self.n_in + 1), order='F')
    
class UORO(Real_Time_Learning_Algorithm):
    
    def __init__(self, net, **kwargs):
        
        allowed_kwargs_ = {'epsilon', 'P0', 'P1'}
        super().__init__(net, allowed_kwargs_, **kwargs)
        
        #Initialize a_tilde and w_tilde vectors
        self.w_tilde = np.random.normal(0, 1, net.n_hidden_params)
        self.a_tilde = np.random.normal(0, 1, net.n_hidden)
        
    def update_learning_vars(self):
        
        #Get relevant values and derivatives from network
        self.a_hat = np.concatenate([self.net.a_prev, self.net.x, np.array([1])])
        self.papw = np.kron(self.a_hat, np.diag(self.net.activation.f_prime(self.net.h)))
        self.net.get_a_jacobian()
        self.nu = np.random.uniform(-1, 1, self.n_h)
        
        #Forward differentiation method
        if hasattr(self, 'epsilon'):
            self.a_eps = self.net.a_prev + self.epsilon*self.a_tilde
            self.f1 = self.net.next_state(self.net.x, self.a_eps, update=False)  
            self.f2 = self.net.next_state(self.net.x, self.net.a_prev, update=False)
            self.a_tilde_ = (self.f1 - self.f2)/self.epsilon
            self.p0 = np.sqrt(norm(self.w_tilde)/(norm(self.a_tilde_) + self.epsilon)) + self.epsilon
            self.p1 = np.sqrt(norm(self.nu.dot(self.papw))/(self.n_h + self.epsilon)) + self.epsilon
        #Backpropagation method
        else:
            self.a_tilde_ = self.net.a_J.dot(self.a_tilde)
            self.p0 = np.sqrt(norm(self.w_tilde)/norm(self.a_tilde_))
            self.p1 = np.sqrt(norm(self.nu.dot(self.papw))/self.n_h)
            
        #Override with fixed P0 and P1 if given
        if hasattr(self, 'P0'):
            self.p0 = np.copy(self.P0)
        if hasattr(self, 'P1'):
            self.p1 = np.copy(self.P1)
            
        #Update outer product approximation
        self.a_tilde = self.p0*self.a_tilde_ + self.p1*self.nu
        self.w_tilde = (1/self.p0)*self.w_tilde + (1/self.p1)*self.nu.dot(self.papw)
    
    def get_rec_grads(self):
        
        self.Q = self.q.dot(self.a_tilde) #"Global learning signal"
        return (self.Q*self.w_tilde).reshape((self.n_h, self.n_h + self.n_in + 1), order='F')
    
class KF_RTRL(Real_Time_Learning_Algorithm):
    
    def __init__(self, net, **kwargs):
        
        allowed_kwargs_ = {'P0', 'P1', 'A', 'u'}
        super().__init__(net, allowed_kwargs_, **kwargs)
        
        #Initialize A and u matrices
        if not hasattr(self, 'A'):
            self.A = np.random.normal(0, 1/np.sqrt(self.n_h), (self.n_h, self.n_h))
        if not hasattr(self, 'u'):
            self.u = np.random.normal(0, 1, self.n_h + self.n_in +1)
        
    def update_learning_vars(self):
        
        #Get relevant values and derivatives from network
        self.a_hat   = np.concatenate([self.net.a_prev, self.net.x, np.array([1])])
        self.D       = np.diag(self.net.activation.f_prime(self.net.h))
        self.net.get_a_jacobian()
        self.H_prime = self.net.a_J.dot(self.A)
        self.c0, self.c1 = np.random.uniform(-1, 1, 2)
        
        #Calculate p0, p1 or override with fixed P0, P1 if given
        if not hasattr(self, 'P0'):
            self.p0 = np.sqrt(norm(self.H_prime)/norm(self.u))
        else:
            self.p0 = np.copy(self.P0)
        if not hasattr(self, 'P1'):
            self.p1 = np.sqrt(norm(self.D)/norm(self.a_hat))
        else:
            self.p1 = np.copy(self.P1)
        
        #Update Kronecker product approximation
        self.u = self.c0*self.p0*self.u + self.c1*self.p1*self.a_hat
        self.A = self.c0*(1/self.p0)*self.H_prime + self.c1*(1/self.p1)*self.D
        
    def get_rec_grads(self):
        
        self.qA = self.q.dot(self.A) #Unit-specific learning signal
        return np.kron(self.u, self.qA).reshape((self.n_h, self.n_h + self.n_in + 1), order='F')

class RFLO(Real_Time_Learning_Algorithm):
    
    def __init__(self, net, alpha, monitors=[], **kwargs):
        
        allowed_kwargs_ = {'P'}
        super().__init__(net, allowed_kwargs_, **kwargs)
        
        self.alpha = alpha
        if not hasattr(self, 'P'):
            self.P = np.zeros((self.n_h, self.n_h + self.n_in + 1))
        
    def update_learning_vars(self):
        
        #Get relevant values and derivatives from network        
        self.a_hat   = np.concatenate([self.net.a_prev, self.net.x, np.array([1])])
        self.D = self.net.activation.f_prime(self.net.h)
        
        #Update eligibility traces
        self.P = (1 - self.alpha)*self.P + self.alpha*np.multiply.outer(self.D, self.a_hat)
        
    def get_rec_grads(self):
        
        return (self.q*self.P.T).T
    
class DNI(Real_Time_Learning_Algorithm):
    
    def __init__(self, net, optimizer, **kwargs):
        
        allowed_kwargs_ = {'SG_clipnorm', 'SG_target_clipnorm', 'W_a_lr',
                           'activation', 'SG_label_activation', 'backprop_weights',
                           'sg_loss_thr', 'U_lr', 'l2_reg', 'fix_SG_interval', 'alpha_e',
                           'train_SG_with_exact_CA'}
        #Default parameters
        self.optimizer = optimizer
        self.l2_reg = 0
        self.fix_SG_interval = 5
        self.activation = identity
        self.SG_label_activation = identity
        self.backprop_weights = 'exact'
        self.sg_loss_thr = 0.05
        self.train_SG_with_exact_CA = False
        #Override defaults with kwargs
        super().__init__(net, allowed_kwargs_, **kwargs)
        
        sigma = np.sqrt(1/self.n_h)
        self.SG_init(sigma)
        
        self.i_fix = 0
        
        self.W_a = np.copy(self.net.W_rec)
        self.U = np.copy(self.A)
        self.A_, self.B_, self.C_ = np.copy(self.A), np.copy(self.B), np.copy(self.C)
        self.SG_params = [self.A, self.B, self.C]
        self.e_w = np.zeros((self.n_h, self.n_h + self.n_in + 1))
        
    def SG_init(self, sigma):
        
        self.A = np.random.normal(0, sigma, (self.n_h, self.n_h))
        self.B = np.random.normal(0, sigma, (self.n_h, self.n_out))
        self.C = np.zeros(self.n_h)
        
    def update_learning_vars(self):
        
        #Get network jacobian
        self.net.get_a_jacobian()
        
        #Computer SG error term
        self.sg = self.synthetic_grad(self.net.a_prev, self.net.y_prev)
        
        if hasattr(self, 'SG_clipnorm'):
            self.sg_norm = norm(self.sg)
            if self.sg_norm > self.SG_clipnorm:
                self.sg = self.sg / self.sg_norm
                
        self.sg_target = self.get_sg_target()
        
        if hasattr(self, 'SG_target_clipnorm'):
            self.sg_target_norm = norm(self.sg_target)
            if self.sg_target_norm > self.SG_target_clipnorm:
                self.sg_target = self.sg_target / self.sg_target_norm
            
        self.e_sg = self.sg - self.sg_target
        self.sg_loss = np.mean((self.sg - self.sg_target)**2)
        self.scaled_e_sg = self.e_sg*self.activation.f_prime(self.sg_h)
        
        #Get SG grads
        self.SG_grads = [np.multiply.outer(self.scaled_e_sg, self.net.a_prev),
                         np.multiply.outer(self.scaled_e_sg, self.net.y_prev),
                         self.scaled_e_sg]
        
        if self.l2_reg > 0:
            self.SG_grads[0] += self.l2_reg*self.A
            self.SG_grads[1] += self.l2_reg*self.B
            self.SG_grads[2] += self.l2_reg*self.C
        
        #Update SG parameters
        self.SG_params = self.optimizer.get_update(self.SG_params, self.SG_grads)
        self.A, self.B, self.C = self.SG_params
        
        if self.i_fix == self.fix_SG_interval - 1:
            self.i_fix = 0
            self.A_, self.B_, self.C_ = np.copy(self.A), np.copy(self.B), np.copy(self.C)
        else:
            self.i_fix += 1
        
        if hasattr(self, 'W_a_lr'):
            self.update_W_a()
            
        if hasattr(self, 'U_lr'):
            self.update_U()
        
    def get_sg_target(self):
        
        self.propagate_feedback_to_hidden()
        
        if self.backprop_weights=='exact':
            sg_target = self.q_prev + self.synthetic_grad_(self.net.a, self.net.y).dot(self.net.a_J)
        elif self.backprop_weights=='approximate':
            sg_target = self.q_prev + self.synthetic_grad_(self.net.a, self.net.y).dot(self.W_a)
        elif self.backprop_weights=='composite':
            sg_target = self.q_prev + self.U.dot(self.net.a)
        
        if self.train_SG_with_exact_CA:
            sg_target = self.net.CA
        
        return sg_target
    
    def update_W_a(self):
        
        self.loss_a = np.square(self.W_a.dot(self.net.a_prev) - self.net.a).mean()
        self.e_a = self.W_a.dot(self.net.a_prev) - self.net.a

        self.W_a -= self.W_a_lr*np.multiply.outer(self.e_a, self.net.a_prev)
        
    def update_U(self):
        
        self.loss_u = np.square(self.U.dot(self.net.a_prev) - self.sg).mean()
        self.e_u = self.U.dot(self.net.a_prev) - self.sg
        self.U -= self.U_lr*np.multiply.outer(self.e_u, self.net.a_prev)
        
    def synthetic_grad(self, a, y):
        self.sg_h = self.A.dot(a) + self.B.dot(y) + self.C
        return self.activation.f(self.sg_h)
        
    def synthetic_grad_(self, a, y):
        self.sg_h_ = self.A_.dot(a) + self.B_.dot(y) + self.C_
        return self.SG_label_activation.f((self.activation.f(self.sg_h_)))
    
    def get_rec_grads(self):
    
        self.sg = self.synthetic_grad(self.net.a, self.net.y)
        self.sg_scaled = self.net.alpha*self.sg*self.net.activation.f_prime(self.net.h)
        
        if hasattr(self, 'SG_clipnorm'):
            sg_norm = norm(self.sg)
            if sg_norm > self.SG_clipnorm:
                self.sg = self.sg / sg_norm
                
        self.a_hat = np.concatenate([self.net.a_prev, self.net.x, np.array([1])])
        
        if hasattr(self, 'alpha_e'):
            self.update_synaptic_eligibility_trace()
            return (self.e_w.T*self.sg).T
        else:
            return np.multiply.outer(self.sg_scaled, self.a_hat)
    
    def update_synaptic_eligibility_trace(self):
        
        self.D = self.net.activation.f_prime(self.net.h)
        self.a_hat = np.concatenate([self.net.a_prev, self.net.x, np.array([1])])
        self.e_w = (1 - self.alpha_e)*self.e_w + self.alpha_e*np.outer(self.D, self.a_hat)

class BPTT(Learning_Algorithm):
    
    def __init__(self, net, t1, t2, monitors=[], use_historical_W=False):
    
        super().__init__(net, monitors)
        
        #The two integer parameters
        self.t1  = t1    #Number of steps to average loss over
        self.t2  = t2    #Number of steps to propagate backwards
        self.T   = t1+t2 #Total amount of net hist needed to memorize
        self.use_historical_W = use_historical_W
        
        #Lists storing relevant net history
        self.h_hist = [np.zeros(self.net.n_hidden) for _ in range(self.T)]
        self.a_hist = [np.zeros(self.net.n_hidden) for _ in range(self.T)]
        self.e_hist = [np.zeros(self.net.n_out) for _ in range(t1)]
        self.x_hist = [np.zeros(self.net.n_in) for _ in range(self.T)]
        self.W_rec_hist = [self.net.W_rec for _ in range(self.T)]
        
    def update_learning_vars(self):
        '''
        Must be run every time step to update storage of relevant history
        '''
        
        for attr in ['h', 'a', 'e', 'x', 'W_rec']:
            self.__dict__[attr+'_hist'].append(getattr(self.net, attr))
            del(self.__dict__[attr+'_hist'][0])
    
    def __call__(self):
        '''
        Run only when grads are needed for the optimizer
        '''
        
        W_out_grad = [np.multiply.outer(self.e_hist[-i],self.a_hist[-i]) for i in range(1, self.t1+1)]
        self.W_out_grad = sum(W_out_grad)/self.t1
        
        b_out_grad = [self.e_hist[-i] for i in range(1, self.t1+1)]
        self.b_out_grad = sum(b_out_grad)/self.t1
        
        self.outer_grads = [self.W_out_grad, self.b_out_grad]
        
        get_a_J = self.net.get_a_jacobian
        
        if self.use_historical_W:
        
            self.a_Js = [get_a_J(update=False,
                         h=self.h_hist[-i],
                         h_prev=self.h_hist[-(i+1)],
                         W_rec=self.W_rec_hist[-i]) for i in range(1, self.T)]
        else:
            
            self.a_Js = [get_a_J(update=False,
                         h=self.h_hist[-i],
                         h_prev=self.h_hist[-(i+1)],
                         W_rec=self.net.W_rec) for i in range(1, self.T)] 
        
        n_h, n_in = self.net.n_hidden, self.net.n_in
        self.rec_grad = np.zeros((n_h, n_h+n_in+1))
        
        CA_list = []
        
        for i in range(1, self.t1+1):
            
            self.q = self.e_hist[-i].dot(self.net.W_out)
            
            for j in range(self.t2):
                
                self.J = np.eye(n_h)
                
                for k in range(j):
                    
                    self.J = self.J.dot(self.a_Js[-(i+k)])
                
                self.J = self.net.alpha*self.J.dot(np.diag(self.net.activation.f_prime(self.h_hist[-(i+j)])))
                
                self.pre_activity = np.concatenate([self.a_hist[-(i+j+1)], self.x_hist[-(i+j)], np.array([1])])
                CA = self.q.dot(self.J)
                CA_list.append(CA) 
                self.rec_grad += np.multiply.outer(CA, self.pre_activity)
        
        self.credit_assignment = sum(CA_list)/len(CA_list)
        
        grads = [self.rec_grad[:,:n_h], self.rec_grad[:,n_h:-1], self.rec_grad[:,-1]]
        grads += self.outer_grads
        
        return grads
    
class Forward_BPTT(Real_Time_Learning_Algorithm):
    
    def __init__(self, net, T, **kwargs):
        
        allowed_kwargs_ = set()
        super().__init__(net, allowed_kwargs_, **kwargs)
        
        self.T = T
        
        self.CA_hist = []
        self.a_hat_hist = []
        self.h_hist = []
        
    def update_learning_vars(self):
        
        #Initialize new credit assignment for current time step
        self.CA_hist.append([0])
        
        #Update history
        self.a_hat = np.concatenate([self.net.a_prev, self.net.x, np.array([1])])
        self.a_hat_hist.append(np.copy(self.a_hat))
        self.h_hist.append(np.copy(self.net.h))
        
        self.propagate_feedback_to_hidden()
        
        for i_CA in range(len(self.CA_hist)):
            
            J = np.eye(self.n_h)
            
            for i_BP in range(i_CA):
            
                J = J.dot(self.net.get_a_jacobian(update=False, h=self.h_hist[-(i_BP+1)]))
                
            self.CA_hist[-(i_CA + 1)] += self.q.dot(J)
                 
    def get_rec_grads(self):
        
        if len(self.CA_hist)==self.T:
            
            self.net.CA = np.copy(self.CA_hist[0])
            
            self.D = self.net.activation.f_prime(self.h_hist[0])
            rec_grads = np.multiply.outer(self.net.CA*self.D, self.a_hat_hist[0])
            
            self.delete_history()
            
        else:
            
            rec_grads = np.zeros((self.n_h, self.n_h + self.n_in + 1))
            
        return rec_grads
            
    def delete_history(self):
        
        for attr in ['CA', 'a_hat', 'h']:
            del(self.__dict__[attr+'_hist'][0])
            
class KeRNL(Real_Time_Learning_Algorithm):
    
    def __init__(self, net, optimizer, T=20, sigma_noise=0.00001,
                 use_approx_kernel=False, **kwargs):

        self.n_h = net.n_hidden
        self.n_in = net.n_in
        self.T = T
        self.sigma_noise = sigma_noise
        self.optimizer = optimizer
        self.zeta = np.random.normal(0, self.sigma_noise, self.n_h)
        
        #Initialize learning variables
        self.beta = np.random.normal(0, 1/np.sqrt(self.n_h), (self.n_h, self.n_h))
        self.gamma = (1/self.T)**np.random.uniform(0, 2, self.n_h)
        self.eligibility = np.zeros((self.n_h, self.n_h + self.n_in + 1))
        self.Omega = np.zeros(self.n_h)
        self.Gamma = np.zeros(self.n_h)
        
        #Initialize noisy network
        self.noisy_net = copy(net)
        
        allowed_kwargs_ = {'beta', 'gamma', 'Omega', 'Gamma', 'eligibility'}
        super().__init__(net, allowed_kwargs_, **kwargs)
        
        if use_approx_kernel:
            self.kernel = self.approx_kernel
        else:
            self.kernel = self.exact_kernel
        
    def exact_kernel(self, delta_t):
        
        return np.exp(-self.gamma*(delta_t))
    
    def approx_kernel(self, delta_t):
        
        return 1 - self.gamma*delta_t
    
    def update_learning_vars(self):
        
        #Observe Jacobian if desired:
        self.J = self.net.get_a_jacobian(update=False)
        
        #Update noisy net's parameters
        self.noisy_net.W_rec = self.net.W_rec
        self.noisy_net.W_in = self.net.W_in
        self.noisy_net.b_rec = self.net.b_rec
        
        #Update noisy net forward
        self.noisy_net.a += self.zeta
        self.noisy_net.next_state(self.net.x)
        
        #Update learning varialbes
        self.zeta = np.random.normal(0, self.sigma_noise, self.n_h)
        self.Gamma = self.kernel(1)*(self.Gamma - self.Omega)
        self.Omega = self.kernel(1)*self.Omega + self.zeta
        
        #Update eligibility traces
        self.D = self.net.activation.f_prime(self.net.h)
        self.a_hat = np.concatenate([self.net.a_prev, self.net.x, np.array([1])])
        self.papw = self.net.alpha*np.multiply.outer(self.D, self.a_hat)
        self.eligibility = (self.eligibility.T*self.kernel(1)).T + self.papw
        
        #Get error in predicting perturbations effect
        self.e_noise = self.beta.dot(self.Omega) - (self.noisy_net.a - self.net.a)
        
        #Update beta and gamma
        self.beta_grads = np.multiply.outer(self.e_noise, self.Omega)
        self.gamma_grads = self.e_noise.dot(self.beta)*self.Gamma
        self.beta, self.gamma = self.optimizer.get_update([self.beta, self.gamma],
                                                          [self.beta_grads, self.gamma_grads])
        
    def get_rec_grads(self):
        
        return (self.eligibility.T*self.q.dot(self.beta)).T








