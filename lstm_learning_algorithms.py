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
from copy import deepcopy
from learning_algorithms import Learning_Algorithm, Stochastic_Algorithm,RTRL


class Only_Output_LSTM(RTRL):
    
    def __init__(self, rnn):
        
        super().__init__(rnn)
    
    def update_learning_vars(self):
        
        pass
    
    def get_rec_grads(self):
        
        dLdw = np.zeros((self.n_h, self.m))
        
        return dLdw

class Stochastic_Algorithm(Learning_Algorithm):

    def sample_nu(self):
        """Sample nu from specified distribution."""

        if self.nu_dist == 'discrete' or self.nu_dist is None:
            nu = np.random.choice([-1, 1], self.n_nu)
        elif self.nu_dist == 'gaussian':
            nu = np.random.normal(0, 1, self.n_nu)
        elif self.nu_dist == 'uniform':
            nu = np.random.uniform(-1, 1, self.n_nu)

        return nu
 

class UORO_LSTM(Stochastic_Algorithm):
    """Implements the Unbiased Online Recurrent Optimization (UORO) algorithm
    from Tallec et al. 2017.

    Full details in our review paper or in original paper. Broadly, an outer
    product approximation of M is maintained by 2 vectors A and B, which update
    by the equations

    A' = p0 J A + p1 \nu        (1)
    B' = 1/p0 B + 1/p1 \nu M_immediate      (2)

    where \nu is a vector of zero-mean iid samples. p0 and p1 are calculated by

    p0 = \sqrt{norm(B)/norm(A)}       (3)
    p1 = \sqrt{norm(\nu papw)/norm(\nu)}        (4)

    These equations are implemented in update_learning_vars by two different
    approaches. If 'epsilon' is provided as an argument, then the "forward
    differentiation" method from the original paper is used, where the matrix-
    vector product JA is estimated numerically by a perturbation of size
    epsilon in the A direction.

    Then the recurrent gradients are calculated by

    dL/dw = qM = (q A) B    (5)

    Eq. (5) is implemented in the get_rec_grads method."""

    def __init__(self, rnn, **kwargs):
        """Inits an UORO instance by setting the initial values of A and B to be
        iid samples from a standard normal distribution, to avoid dividing by
        zero in Eqs. (3) and (4).

        Keyword args:
            epsilon (float): Scaling factor on perturbation for forward
                differentiation method. If not provided, exact derivative is
                calculated instead.
            P0 (float): Overrides calculation of p0, instead uses provided value
                of P0. If not provided, p0 is calculated according to Eq. (3).
            P1 (float): Same for p1.
            A (numpy array): Initial value for A.
            B (numpy array): Initial value for B.
            nu_dist (string): Takes on the value of 'gaussian', 'discrete', or
                'uniform' to indicate what type of distribution nu should sample
                 from. Default is 'discrete'."""

        self.name = 'UORO' #Default algorithm name
        allowed_kwargs_ = {'epsilon', 'P0', 'P1','A','B', 'A_f', 'B_f','A_i', 'B_i',
        'A_a', 'B_a','A_o', 'B_o', 'nu_dist'}
        super().__init__(rnn, allowed_kwargs_, **kwargs)
        self.n_nu = self.n_t 

        #Initialize A and B arrays
        if self.A is None:
            self.A = np.random.normal(0, 1, self.n_t) 
        if self.B is None:
            self.B = np.random.normal(0, 1, (self.n_h , self.m)) 
            
    def update_learning_vars(self, update=True):
        """Implements Eqs. (1), (2), (3), and (4) to update the outer product
        approximation of the influence matrix by A and B.

        Args:
            update (bool): If True, updates the algorithm's current outer
                product approximation B, A. If False, only prepares for calling
                get_influence_estimate."""

        self.rnn.get_a_jacobian() #Get updated network Jacobian
        self.rnn.update_compact_M_immediate() #Get immediate influence papw
        

        A, B = self.get_influence_estimate()

        if update:
            self.A, self.B = A, B


    def get_influence_estimate(self):
        """Generates one random outer-product estimate of the influence matrix.

        Samples a random vector nu of iid samples with 0 mean from a
        distribution given by nu_dist, and returns an updated estimate
        of A and B from Eqs. (1)-(4).

        Returns:
            Updated A (numpy array of shape (n_h)) and B (numpy array of shape
                (n_h, m))."""

        #Sample random vector
        self.nu = self.sample_nu()
        
        #Get random projection of M_immediate onto \nu
        #M_projection = self.nu.dot(self.rnn.papw)
        M_projection_o = (self.rnn.papw_c.T*self.nu).T
        M_projection = M_projection_o[:self.n_h,:] + M_projection_o[self.n_h:,:]
        


        if self.epsilon is not None: #Forward differentiation method
            eps = self.epsilon
            #Get perturbed state in direction of A
            self.c_perturbed = self.rnn.c_prev + eps * self.A[:self.n_t]
            self.h_perturbed = self.rnn.h_prev + eps * self.A[self.n_t:]
            #Get hypothetical next states from this perturbation
            self.a_perturbed_next = self.rnn.next_state(self.rnn.x,
                                                        self.c_perturbed,
                                                        self.h_preturbed,
                                                        update=False)
            #Get forward-propagated A
            c_h = np.concatenate([self.rnn.c, self.rnn.h])
            self.A_forwards = (self.a_perturbed_next - c_h) / eps
            #Calculate scaling factors
            B_norm = norm(self.B)
            A_norm = norm(self.A_forwards)
            M_norm = norm(M_projection)
            self.p0 = np.sqrt(B_norm/(A_norm + eps)) + eps
            self.p1 = np.sqrt(M_norm/(np.sqrt(self.n_t) + eps)) + eps
        else: #Backpropagation method
            #Get forward-propagated A
            self.A_forwards = self.rnn.a_J.dot(self.A)
            #Calculate scaling factors
            B_norm = norm(self.B)
            A_norm = norm(self.A_forwards)
            M_norm = norm(M_projection)
            nu_norm = norm(self.nu)
            self.p0 = np.sqrt(B_norm/A_norm)

            self.p1 = np.sqrt(M_norm/np.sqrt(self.n_t)) #np.sqrt(self.n_t)

 
            if self.p1 == 0:
                print('M',M_norm,self.rnn.papw_c)



        #Override with fixed P0 and P1 if given
        if self.P0 is not None:
            self.p0 = np.copy(self.P0)
        if self.P1 is not None:
            self.p1 = np.copy(self.P1)

        #Update outer product approximation
        A = self.p0 * self.A_forwards + self.p1 * self.nu
        B = (1/self.p0) * self.B + (1 / self.p1) * M_projection
        #print('p0',self.p0)
        return A, B



    def get_rec_grads(self):
        """Calculates recurrent grads by taking matrix product of q with the
        estimate of the influence matrix.

        First associates q with A to calculate a "global learning signal"
        Q, which multiplies by B to compute the recurrent gradient, which
        is reshaped into original matrix form.

        Returns:
            An array of shape (n_h, m) representing the recurrent gradient."""

        #dLdw = self.q.dot(np.outer(self.A,self.B)).reshape((self.n_h, self.m), order='F')
        dLdw = self.q.dot(self.A) * self.B
    
        return dLdw

    def reset_learning(self):
        """Resets learning by re-randomizing the outer product approximation to
        random gaussian samples."""

        self.A = np.random.normal(0, 1, self.n_t) 
        self.B = np.random.normal(0, 1, (self.n_h , self.m)) 
    


class KF_RTRL_LSTM(Stochastic_Algorithm):
    """Implements the Kronecker-Factored Real-Time Recurrent Learning Algorithm
    (KF-RTRL) from Mujika et al. 2018.

    Details in review paper or original Mujika et al. 2018. Broadly, M is
    approximated as a Kronecker product between a (row) vector A and a matrix
    B, which updates as

    A' = \nu_0 p0 A + \nu_1 p1 a_hat        (1)
    B' = \nu_0 1/p0 JB + \nu_1 1/p1 \alpha diag(\phi'(h))      (2)

    where \nu = (\nu_0, \nu_1) is a vector of zero-mean iid samples, a_hat is
    the concatenation [a_prev, x, 1], and p0 and p1 are calculated by

    p0 = \sqrt{norm(JB)/norm(A)}       (3)
    p1 = \sqrt{norm(D)/norm(a_hat)}        (4)

    Then the recurrent gradients are calculated by

    dL/dw = qM = A (qB)    (5)

    Eq. (5) is implemented in the get_rec_grads method.
    """

    def __init__(self, rnn, **kwargs):
        """Inits a KF-RTRL instance by setting the initial values of A and B to
        be iid samples from a gaussian distributions, to avoid dividing by
        zero in Eqs. (3) and (4).

        Keyword args:
            P0 (float): Overrides calculation of p0, instead uses provided value
                of P0. If not provided, p0 is calculated according to Eq. (3).
            P1 (float): Same for p1.
            A (numpy array): Initial value for A.
            B (numpy array): Initial value for B.
            nu_dist (string): Takes on the value of 'gaussian', 'discrete', or
                'uniform' to indicate what type of distribution nu should sample
                 from. Default is 'discrete'."""

        self.name = 'KF-RTRL'
        allowed_kwargs_ = {'P0', 'P1', 'A', 'B', 'nu_dist'}
        super().__init__(rnn, allowed_kwargs_, **kwargs)
        self.n_nu = 2

        #Initialize A and B arrays
        if self.A is None:
            self.A = np.random.normal(0, 1, self.n_in+self.n_h+1)
        if self.B is None:
            self.B = np.random.normal(0, 1, (self.n_t, self.n_h*4))

    def update_learning_vars(self, update=True):
        """Implements Eqs. (1), (2), (3), and (4) to update the Kron. product
        approximation of the influence matrix by A and B.

        Args:
            update (bool): If True, updates the algorithm's current outer
                product approximation B, A. If False, only prepares for calling
                get_influence_estimate."""

        #Get relevant values and derivatives from network
        self.a_hat = np.concatenate([self.rnn.h_hat_prev,
                                     np.array([1])])

        r = self.rnn.o * self.rnn.tanh.f_prime(self.rnn.c)

        D_o_c = np.zeros((self.n_h, self.n_h))
        D_o_h = np.diag((self.rnn.o-self.rnn.o**2) * self.rnn.tanh.f(self.rnn.c))
        D_o = np.concatenate([D_o_c, D_o_h])


        D_a = np.diag((1-self.rnn.a**2) * self.rnn.i)
        D_i = np.diag((self.rnn.i-self.rnn.i**2) * self.rnn.a)
        D_f = np.diag((self.rnn.f-self.rnn.f**2) * self.rnn.c_prev)


        D_c = np.concatenate([D_f,D_i,D_a],axis=1)
        D_h = (D_c.T*r).T
        self.D = np.concatenate([np.concatenate([D_c,D_h]),D_o],axis=1)


        self.rnn.get_a_jacobian()
        self.B_forwards = self.rnn.a_J.dot(self.B)

        A, B = self.get_influence_estimate()
        
        # test1 = np.kron(self.a_hat,self.D[:,:32])
        # test2 = np.kron(self.a_hat,self.D[:,32:64])
        # test3 = np.kron(self.a_hat,self.D[:,64:96])
        # test4 = np.kron(self.a_hat,self.D[:,96:128])
        # test = np.concatenate([test1,test2,test3,test4],axis=1)
        # papw = self.rnn.update_M_immediate(update=False)
        
        # if np.isclose(test, papw).all():
        #     print('True')
        # else:
        #     print('False')


        if update:
            self.A, self.B = A, B

    def get_influence_estimate(self):
        """Generates one random Kron.-product estimate of the influence matrix.

        Samples a random vector nu of iid samples with 0 mean from a
        distribution given by nu_dist, and returns an updated estimate
        of A and B from Eqs. (1)-(4).

        Returns:
            Updated A (numpy array of shape (m)) and B (numpy array of shape
                (n_h, n_h))."""

        #Sample random vector (shape (2) in KF-RTRL)
        self.nu = self.sample_nu()

        #Calculate p0, p1 or override with fixed P0, P1 if given
        if self.P0 is None:
            self.p0 = np.sqrt(norm(self.B_forwards)/norm(self.A))
        else:
            self.p0 = np.copy(self.P0)
        if self.P1 is None:
            self.p1 = np.sqrt(norm(self.D)/norm(self.a_hat))
        else:
            self.p1 = np.copy(self.P1)

        #Update Kronecker product approximation
        A = self.nu[0]*self.p0*self.A + self.nu[1]*self.p1*self.a_hat
        B = (self.nu[0]*(1/self.p0)*self.B_forwards +
             self.nu[1]*(1/self.p1)*self.D)

        return A, B

    def get_rec_grads(self):
        """Calculates recurrent grads by taking matrix product of q with the
        estimate of the influence matrix.

        First associates q with B to calculate a vector qB, whose Kron. product
        with A (effectively an outer product upon reshaping) gives the estimated
        recurrent gradient.

        Returns:
            An array of shape (n_h, m) representing the recurrent gradient."""

        y = self.B.shape[1]//4
        B1,B2,B3,B4 = self.B[:,:y], self.B[:,y:2*y],self.B[:,2*y:3*y],self.B[:,3*y:4*y]
        p1 = np.kron(self.A, B1)
        p2 = np.kron(self.A, B2)
        p3 = np.kron(self.A, B3)
        p4 = np.kron(self.A, B4)
        #print(p4.shape)
        p = np.concatenate([p1,p2,p3,p4],axis=1)
        return self.q.dot(p).reshape((self.n_h, self.m), order='F')

    def reset_learning(self):
        """Resets learning by re-randomizing the outer product approximation to
        random gaussian samples."""

        self.A = np.random.normal(0, 1, self.n_h+self.n_in+1)
        self.B = np.random.normal(0, 1, (self.n_t, self.n_h*4))


class Efficient_BPTT_LSTM(Learning_Algorithm):
    """Implements the 'E-BPTT' version of backprop we discuss in the paper for
    an RNN.

    We describe in more detail in the paper. In brief, the network activity is
    'unrolled' for T_trunction time steps in non-overlapping intervals. The
    gradient for each interval is computed using the future-facing relation
    from Section 2. Thus 'update_learning_vars' is called at every step to
    update the memory of relevant network variables, while get_rec_grads only
    returns non-zero elements every T_truncation time steps."""

    def __init__(self, rnn, T_truncation, **kwargs):
        """Inits an instance of Efficient_BPTT by specifying the network to
        train and the truncation horizon. No default allowable kwargs."""

        self.name = 'E-BPTT'
        allowed_kwargs_ = set()
        super().__init__(rnn, allowed_kwargs_, **kwargs)

        self.T_truncation = T_truncation

        # Initialize lists for storing network data
        self.a_hat_history = []
        self.h_history = []
        self.c_history = []
        self.c_prev_history = []
        self.q_history = []

    def update_learning_vars(self):
        """Updates the memory of the algorithm with the relevant network
        variables for running E-BPTT."""

        # Add latest values to list
        self.a_hat_history.insert(0, np.concatenate([self.rnn.h_hat_prev,                                                
                                                     np.array([1])]))
        self.h_history.insert(0, self.rnn.h)
        self.c_history.insert(0, self.rnn.c)
        self.c_prev_history.insert(0, self.rnn.c_prev)
        self.propagate_feedback_to_hidden()
        self.q_history.insert(0, self.q)

    def get_rec_grads(self):
        """Using the accumulated history of q, h and a_hat values over the
        truncation interval, computes the recurrent gradient.

        Returns:
            rec_grads (numpy array): Array of shape (n_h, m) representing
                the gradient dL/dW after truncation interval completed,
                otherwise an array of 0s of the same shape."""

        #Once a 'triangle' is formed (see Fig. 3 in paper), compute gradient.
        if len(self.a_hat_history) >= self.T_truncation:

            #Initialize recurrent grads at 0
            rec_grads = np.zeros((self.n_h, self.m))
            #Start with most recent credit assignment value
            credit = self.q_history.pop(0)
            #print('credit', credit.shape)
            for i_BPTT in range(self.T_truncation):

                # Access present values of h and a_hat
                h = self.h_history.pop(0)
                c = self.c_history.pop(0)
                a_hat = self.a_hat_history.pop(0)
                c_prev = self.c_prev_history.pop(0)

                #Use to get gradients w.r.t. weights from credit assignment
                papw = self.rnn.update_M_immediate(c=c, c_prev = c_prev, state_hat=a_hat, update=False)
                # print('papw',papw.shape)
                # print(credit.dot(papw).reshape((self.n_h, self.m), order='F').shape)
                rec_grads += credit.dot(papw).reshape((self.n_h, self.m), order='F')
                
                if i_BPTT == self.T_truncation - 1: #Skip if at end
                    continue

                #Use future-facing relation to backpropagate by one time step.
                q = self.q_history.pop(0)
                J = self.rnn.get_a_jacobian(h=h, c=c, update=False)
                credit = q + credit.dot(J)

            return rec_grads

        else:

            return np.zeros((self.n_h, self.m))


class Future_BPTT_LSTM(Learning_Algorithm):
    """Implements the 'F-BPTT' version of backprop we discuss in the paper for
    an RNN.

    Although more expensive than E-BPTT by a factor of the truncation horizon,
    this version covers more 'loss-parmaeter sensitivity' terms in Fig. 3 and
    produces, at each time step, an approximate 'future-facing' gradient up to
    truncation that can be used for comparison with other algorithm's outputs.

    Details of computation are in paper. When a credit assignment estimate is
    calculated, the gradient is ultimately calculated according to

    dL/dW_{ij} = c_i \phi'(h_i) a_hat_j                             (1)."""

    def __init__(self, rnn, T_truncation, **kwargs):
        """Inits an instance of Future_BPTT by specifying the network to
        train and the truncation horizon. No default allowable kwargs."""

        self.name = 'F-BPTT'
        allowed_kwargs_ = set()
        super().__init__(rnn, allowed_kwargs_, **kwargs)

        self.T_truncation = T_truncation
        
        self.a_hat_history = []
        self.q_history = []
        self.c_history = []
        self.c_prev_history = []
        self.h_history = []       
        

    def update_learning_vars(self):
        """Updates the list of credit assignment vectors according to Section
        4.1.2 in the paper.

        First updates relevant history with latest network variables
        a_hat, h and q. Then backpropagates the latest q to each previous time
        step, adding the result to each previous credit assignment estimate."""

        #Update history
        self.a_hat_history.insert(0, np.concatenate([self.rnn.h_hat_prev,
                                                     np.array([1])]))
        self.h_history.insert(0, self.rnn.h)
        self.c_history.insert(0, self.rnn.c)
        self.c_prev_history.insert(0, self.rnn.c_prev)
        self.propagate_feedback_to_hidden()
        q = np.copy(self.q)
        #Add immediate credit assignment to front of list
        self.q_history.insert(0, q)

        #Loop over truncation horizon and backpropagate q, pausing along way to
        #update credit assignment estimates
        for i_BPTT in range(1, len(self.q_history)):

            h = self.h_history[i_BPTT - 1]
            c = self.c_history[i_BPTT - 1]
            J = self.rnn.get_a_jacobian(h=h, c=c, update=False)
            q = q.dot(J)
            self.q_history[i_BPTT] += q

    def get_rec_grads(self):
        """Removes the oldest credit assignment value from the c_history list
        and uses it to produce recurrent gradients according to Eq. (1).

        Note: for the first several time steps of the simulation, before
        self.c_history fills up to T_truncation size, 0s are returned for
        the recurrent gradients."""

        if len(self.q_history) >= self.T_truncation:

            #Remove oldest c, h and a_hat from lists
            c = self.c_history.pop(-1)
            c_prev = self.c_prev_history.pop(-1)
            h = self.h_history.pop(-1)
            q = self.q_history.pop(-1)
            a_hat = self.a_hat_history.pop(-1)

            #Implement Eq. (1)
            papw = self.rnn.update_M_immediate(c=c, c_prev = c_prev, state_hat=a_hat, update=False)
            rec_grads = q.dot(papw).reshape((self.n_h, self.m), order='F')

        else:

            rec_grads = np.zeros((self.n_h, self.m))

        return rec_grads

    def reset_learning(self):
        """Resets learning by deleting network variable history."""

        self.c_history = []
        self.c_prev_history = []
        self.a_hat_history = []
        self.h_history = []
        self.q_history = []