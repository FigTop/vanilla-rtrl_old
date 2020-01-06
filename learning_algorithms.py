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
    """Parent class for all types of learning algorithms.

    Attributes:
        net (network.RNN): An instance of RNN to be trained by the network.
        n_* (int): Extra pointer to net.n_* (in, h, out) for conveneince.
        m (int): Number of recurrent "input dimensions" n_h + n_in + 1 including
            task inputs and constant 1 for bias.
        q (numpy array): Array of immediate error signals for the hidden units,
            i.e. the derivative of the current loss with respect to net.a, of
            shape (n_h).
        W_FB (numpy array or None): A fixed set of weights that may be provided
            for an approximate calculation of q in the manner of feedback
            alignment (Lillicrap et al. 2016).
        SG_L2_reg (float or None): Strength of L2 regularization parameter on the
            network weights."""

    def __init__(self, net, allowed_kwargs_=set(), **kwargs):
        """Initializes an instance of learning algorithm by specifying the
        network to be trained, custom allowable kwargs, and kwargs.

        Args:
            net (network.RNN): An instance of RNN to be trained by the network.
            allowed_kwargs_ (set): Set of allowed kwargs in addition to those
                common to all child classes of Learning_Algorithm, 'W_FB' and
                'L2_reg'."""

        allowed_kwargs = {'W_FB', 'L2_reg'}.union(allowed_kwargs_)

        for k in kwargs:
            if k not in allowed_kwargs:
                raise TypeError('Unexpected keyword argument passed'
                                'to Learning_Algorithm.__init__: ' + str(k))

        #Set all non-specified kwargs to None
        for attr in allowed_kwargs:
            if not hasattr(self, attr):
                setattr(self, attr, None)

        #Make kwargs attributes of the instance
        self.__dict__.update(kwargs)

        #Define basic learning algorithm properties
        self.net = net
        self.n_in = self.net.n_in
        self.n_h = self.net.n_h
        self.n_out = self.net.n_out
        self.m = self.n_h + self.n_in + 1
        self.q = np.zeros(self.n_h)

    def reset_learning(self):
        """Resets internal variables of the learning algorithm (relevant if
        simulation includes a trial structure). Default is to do nothing."""

        pass

class Real_Time_Learning_Algorithm(Learning_Algorithm):
    """Parent class for all learning algorithms that run "in real time," i.e.
    collect and apply errors from the task as they occur.

    Attributes:
        a_ (numpy array): Array of shape (n_h + 1) that is the concatenation of
            the network's state and the constant 1, used to calculate the output
            errors.
        q_prev (numpy array): The q value from the previous time step."""

    def get_outer_grads(self):
        """Calculates the derivative of the loss with respect to the output
        parameters net.W_out and net.b_out.

        Calculates the outer gradients in the manner of a perceptron derivative
        by taking the outer product of the error with the "regressors" onto the
        output (the hidden state and constant 1).

        Returns:
            A numpy array of shape (net.n_out, self.n_h + 1) containing the
                concatenation (along column axis) of the derivative of the loss
                w.r.t. net.W_out and w.r.t. net.b_out."""

        self.a_ = np.concatenate([self.net.a, np.array([1])])
        return np.multiply.outer(self.net.error, self.a_)

    def propagate_feedback_to_hidden(self):
        """Performs one step of backpropagation from the outer-layer errors to
        the hidden state.

        Calculates the immediate derivative of the loss with respect to the
        hidden state net.a. By default, this is done by taking net.error (dL/dz)
        and applying the chain rule, i.e. taking its matrix product with the
        derivative dz/da, which is net.W_out. Alternatively, if 'W_FB' attr is
        provided to the instance, then these feedback weights, rather the W_out,
        are used, as in feedback alignment. (See Lillicrap et al. 2016.)

        Updates q to the current value of dL/da."""

        self.q_prev = np.copy(self.q)

        if self.W_FB is None:
            self.q = self.net.error.dot(self.net.W_out)
        else:
            self.q = self.net.error.dot(self.W_FB)

    def SG_L2_regularization(self, grads):
        """Adds L2 regularization to the gradient.

        Args:
            grads (list): List of numpy arrays representing gradients before L2
                regularization is applied.
        Returns:
            A new list of grads with L2 regularization applied."""

        #Get parameters affected by L2 regularization
        L2_params = [self.net.params[i] for i in self.net.L2_indices]
        #Add to each grad the corresponding weight's current value, weighted
        #by the SG_L2_reg hyperparameter.
        for i_L2, W in zip(self.net.L2_indices, L2_params):
            grads[i_L2] += self.SG_L2_reg * W
        #Calculate L2 loss for monitoring purposes
        self.L2_loss = 0.5*sum([norm(p) for p in L2_params])
        return grads

    def __call__(self):
        """Calculates the final list of grads for this time step.

        Assumes the user has already called self.update_learning_vars, a
        method specific to each child class of Real_Time_Learning_Algorithm
        that updates internal learning variables, e.g. the influence matrix of
        RTRL. Then calculates the outer grads (gradients of W_out and b_out),
        updates q using propagate_feedback_to_hidden, and finally calling the
        get_rec_grads method (specific to each child class) to get the gradients
        of W_rec, W_in, and b_rec as one numpy array with shape (n_h, m). Then
        these gradients are split along the column axis into a list of 5
        gradients for W_rec, W_in, b_rec, W_out, b_out. L2 regularization is
        applied if SG_L2_reg parameter is not None.

        Returns:
            List of gradients for W_rec, W_in, b_rec, W_out, b_out."""

        self.outer_grads = self.get_outer_grads()
        self.propagate_feedback_to_hidden()
        self.rec_grads = self.get_rec_grads()
        rec_grads_list = split_weight_matrix(self.rec_grads,
                                             [self.n_h, self.n_in, 1])
        outer_grads_list = split_weight_matrix(self.outer_grads,
                                               [self.n_h, 1])
        grads_list = rec_grads_list + outer_grads_list

        if self.SG_L2_reg is not None:
            grads_list = self.SG_L2_regularization(grads_list)

        return grads_list

class Only_Output_Weights(Real_Time_Learning_Algorithm):
    """Updates only the output weights W_out and b_out"""

    def __init__(self, net, **kwargs):

        self.name = 'Only_Output_Weights'
        allowed_kwargs_ = set()
        super().__init__(net, allowed_kwargs_, **kwargs)

    def update_learning_vars(self):
        """No internal variables to update."""

        pass

    def get_rec_grads(self):
        """Returns all 0s for the recurrent gradients."""

        return np.zeros((self.n_h, self.m))

class RTRL(Real_Time_Learning_Algorithm):
    """Implements the Real-Time Recurrent Learning (RTRL) algorithm.

    RTRL maintains a long-term "influence matrix" dadw that represents the
    derivative of the hidden state with respect to a flattened vector of
    recurrent update parameters. We concatenate [W_rec, W_in, b_rec] along
    the column axis and order the flattened vector of parameters by stacking
    the columns end-to-end. In other words, w_k = W_{ij} when i = k%n_h and
    j = k//n_h. The influence matrix updates according to the equation

    M' = JM + M_immediate       (1)

    where J is the network Jacobian and M_immediate is the immediate influence
    of a parameter w on the hidden state a. (See paper for more detailed
    notation.) M_immediate is notated as papw in the code for "partial a partial
    w." For a vanilla network, this can be simply (if inefficiently) computed as
    the Kronecker product of a_hat = [a_prev, x, 1] (a concatenation of the prev
    hidden state, the input, and a constant 1 (for bias)) with the activation
    derivatives organized in a diagonal matrix. The implementation of Eq. (1)
    is in the update_learning_vars method.

    Finally, the algorithm returns recurrent gradients by projecting the
    feedback vector q onto the influence matrix M:

    dL/dw = dL/da da/dw = qM        (2)

    Eq. (2) is implemented in the get_rec_grads method."""

    def __init__(self, net, **kwargs):
        """Inits an RTRL instance by setting the initial dadw matrix to zero."""

        self.name = 'RTRL' #Algorithm name
        allowed_kwargs_ = set() #No special kwargs for RTRL
        super().__init__(net, allowed_kwargs_, **kwargs)

        #Initialize influence matrix
        self.dadw = np.zeros((self.n_h, self.net.n_h_params))

    def update_learning_vars(self):
        """Updates the influence matrix via Eq. (1)."""

        #Get relevant values and derivatives from network.
        self.a_hat = np.concatenate([self.net.a_prev,
                                     self.net.x,
                                     np.array([1])])
        D = np.diag(self.net.activation.f_prime(self.net.h))
        self.papw = np.kron(self.a_hat, D) #Calculate M_immediate
        self.net.get_a_jacobian() #Get updated network Jacobian

        #Update influence matrix via Eq. (1).
        self.dadw = self.net.a_J.dot(self.dadw) + self.papw

    def get_rec_grads(self):
        """Calculates recurrent grads using Eq. (2), reshapes into original
        matrix form."""

        return self.q.dot(self.dadw).reshape((self.n_h, self.m), order='F')

    def reset_learning(self):
        """Resets learning algorithm by setting influence matrix to 0."""

        self.dadw *= 0

class UORO(Real_Time_Learning_Algorithm):
    """Implements the Unbiased Online Recurrent Optimization (UORO) algorithm.

    Full details in our review paper or in original Tallec et al. 2017. Broadly,
    an outer product approximation of M is maintained by 2 vectors A and B,
    which update by the equations

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

    def __init__(self, net, **kwargs):
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
        allowed_kwargs_ = {'epsilon', 'P0', 'P1', 'A', 'B', 'nu_dist'}
        super().__init__(net, allowed_kwargs_, **kwargs)

        #Initialize A and B arrays
        if self.A is None:
            self.A = np.random.normal(0, 1, self.n_h)
        if self.B is None:
            self.B = np.random.normal(0, 1, (self.n_h, self.m))

    def update_learning_vars(self, update=True):
        """Implements Eqs. (1), (2), (3), and (4) to update the outer product
        approximation of the influence matrix by A and B.

        Args:
            update (bool): If True, updates the algorithm's current outer
                product approximation B, A. If False, only prepares for calling
                get_influence_estimate."""

        #Get relevant values and derivatives from network
        self.a_hat = np.concatenate([self.net.a_prev,
                                     self.net.x,
                                     np.array([1])])
        D = self.net.activation.f_prime(self.net.h)
        #Compact form of M_immediate
        self.papw = np.multiply.outer(D, self.a_hat)
        self.net.get_a_jacobian() #Get updated network Jacobian

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

        #Sample nu from specified distribution
        if self.nu_dist == 'discrete' or self.nu_dist is None:
            self.nu = np.random.choice([-1, 1], self.n_h)
        elif self.nu_dist == 'gaussian':
            self.nu = np.random.normal(0, 1, self.n_h)
        elif self.nu_dist == 'uniform':
            self.nu = np.random.uniform(-1, 1, self.n_h)

        #Get random projection of M_immediate onto \nu
        M_projection = (self.papw.T*self.nu).T

        if self.epsilon is not None: #Forward differentiation method
            eps = self.epsilon
            #Get perturbed state in direction of A
            self.a_perturbed = self.net.a_prev + eps * self.A
            #Get hypothetical next states from this perturbation
            self.a_perturbed_next = self.net.next_state(self.net.x,
                                                        self.a_perturbed,
                                                        update=False)
            #Get forward-propagated A
            self.A_forwards = (self.a_perturbed_next - self.net.a)/eps
            #Calculate scaling factors
            B_norm = norm(self.B)
            A_norm = norm(self.A_forwards)
            M_norm = norm(M_projection)
            self.p0 = np.sqrt(B_norm/(A_norm + eps)) + eps
            self.p1 = np.sqrt(M_norm/(np.sqrt(self.n_h) + eps)) + eps
        else: #Backpropagation method
            #Get forward-propagated A
            self.A_forwards = self.net.a_J.dot(self.A)
            #Calculate scaling factors
            B_norm = norm(self.B)
            A_norm = norm(self.A_forwards)
            M_norm = norm(M_projection)
            self.p0 = np.sqrt(B_norm/A_norm)
            self.p1 = np.sqrt(M_norm/np.sqrt(self.n_h))

        #Override with fixed P0 and P1 if given
        if self.P0 is not None:
            self.p0 = np.copy(self.P0)
        if self.P1 is not None:
            self.p1 = np.copy(self.P1)

        #Update outer product approximation
        A = self.p0 * self.A_forwards + self.p1 * self.nu
        B = (1/self.p0) * self.B + (1 / self.p1) * M_projection

        return A, B

    def get_rec_grads(self):
        """Calculates recurrent grads by taking matrix product of q with the
        estimate of the influence matrix.

        First associates q with A to calculate a "global learning signal"
        Q, which multiplies by B to compute the recurrent gradient, which
        is reshaped into original matrix form.

        Returns:
            An array of shape (n_h, m) representing the recurrent gradient."""

        self.Q = self.q.dot(self.A) #"Global learning signal"
        return (self.Q * self.B)

    def reset_learning(self):
        """Resets learning by re-randomizing the outer product approximation to
        random gaussian samples."""

        self.A = np.random.normal(0, 1, self.n_h)
        self.B = np.random.normal(0, 1, (self.n_h, self.m))

class KF_RTRL(Real_Time_Learning_Algorithm):
    """Implements the Kronecker-Factored Real-Time Recurrent Learning Algorithm
    (KF-RTRL).

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

    def __init__(self, net, **kwargs):
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
        super().__init__(net, allowed_kwargs_, **kwargs)

        #Initialize A and B arrays
        if self.A is None:
            self.A = np.random.normal(0, 1, self.m)
        if self.B is None:
            self.B = np.random.normal(0, 1/np.sqrt(self.n_h),
                                      (self.n_h, self.n_h))

    def update_learning_vars(self, update=True):
        """Implements Eqs. (1), (2), (3), and (4) to update the Kron. product
        approximation of the influence matrix by A and B.

        Args:
            update (bool): If True, updates the algorithm's current outer
                product approximation B, A. If False, only prepares for calling
                get_influence_estimate."""

        #Get relevant values and derivatives from network
        self.a_hat   = np.concatenate([self.net.a_prev, self.net.x, np.array([1])])
        self.D       = np.diag(self.net.activation.f_prime(self.net.h))
        self.net.get_a_jacobian()
        self.B_forwards = self.net.a_J.dot(self.B)

        A, B = self.get_influence_estimate()

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

        #Sample nu from specified distribution
        if self.nu_dist == 'discrete' or self.nu_dist is None:
            self.nu = np.random.choice([-1, 1], 2)
        elif self.nu_dist == 'gaussian':
            self.nu = np.random.normal(0, 1, 2)
        elif self.nu_dist == 'uniform':
            self.nu = np.random.uniform(-1, 1, 2)

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

        self.qB = self.q.dot(self.B) #Unit-specific learning signal
        return np.kron(self.A, self.qB).reshape((self.n_h, self.m), order='F')

    def reset_learning(self):
        """Resets learning by re-randomizing the outer product approximation to
        random gaussian samples."""

        self.A = np.random.normal(0, 1, self.m)
        self.B = np.random.normal(0, 1/np.sqrt(self.n_h),
                                  (self.n_h, self.n_h))

class Reverse_KF_RTRL(Real_Time_Learning_Algorithm):
    """Implements the "Reverse" KF-RTRL (R-KF-RTRL) algorithm.

    Full details in our review paper. Broadly, an approximation of M in the form
    of a Kronecker product between a matrix B and a (row) vector A is maintained
    by the update

    A'_i = p0 A_i + p1 \nu_i        (1)
    B'_{kj} = (1/p0 \sum_{k'} J+{kk'}B_{k'j} +
               1/p1 \sum_i \nu_i M_immediate_{kij})      (2)

    where \nu is a vector of zero-mean iid samples. p0 and p1 are calculated by

    p0 = \sqrt{norm(B)/norm(A)}       (3)
    p1 = \sqrt{norm(\nu papw)/norm(\nu)}        (4)

    Then the recurrent gradients are calculated by

    dL/dw = qM = A (qB)    (5)

    Eq. (5) is implemented in the get_rec_grads method."""

    def __init__(self, net, **kwargs):
        """Inits an R-KF-RTRL instance by setting the initial values of A and B
        to be iid samples from gaussian distributions, to avoid dividing by
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

        self.name = 'R-KF-RTRL'
        allowed_kwargs_ = {'P0', 'P1', 'A', 'B', 'nu_dist'}
        super().__init__(net, allowed_kwargs_, **kwargs)

        #Initialize A and B arrays
        if self.A is None:
            self.A = np.random.normal(0, 1, self.n_h)
        if self.B is None:
            self.B = np.random.normal(0, 1/np.sqrt(self.n_h),
                                      (self.n_h, self.m))

    def update_learning_vars(self, update=True):
        """Implements Eqs. (1), (2), (3), and (4) to update the Kron. product
        approximation of the influence matrix by A and B.

        Args:
            update (bool): If True, updates the algorithm's current outer
                product approximation B, A. If False, only prepares for calling
                get_influence_estimate."""

        #Get relevant values and derivatives from network
        self.a_hat = np.concatenate([self.net.a_prev,
                                     self.net.x,
                                     np.array([1])])
        self.D = self.net.activation.f_prime(self.net.h)
        #Compact form of M_immediate
        self.papw = np.multiply.outer(self.D, self.a_hat)
        self.net.get_a_jacobian() #Get updated network Jacobian
        self.B_forwards = self.net.a_J.dot(self.B)

        A, B = self.get_influence_estimate()

        if update:
            self.A, self.B = A, B

    def get_influence_estimate(self):
        """Generates one random Kron.-product estimate of the influence matrix.

        Samples a random vector nu of iid samples with 0 mean from a
        distribution given by nu_dist, and returns an updated estimate
        of A and B from Eqs. (1)-(4).

        Returns:
            Updated A (numpy array of shape (n_h)) and B (numpy array of shape
                (n_h, n_m))."""

        #Sample nu from specified distribution
        if self.nu_dist == 'discrete' or self.nu_dist is None:
            self.nu = np.random.choice([-1, 1], self.n_h)
        elif self.nu_dist == 'gaussian':
            self.nu = np.random.normal(0, 1, self.n_h)
        elif self.nu_dist == 'uniform':
            self.nu = np.random.uniform(-1, 1, self.n_h)

        # Get random projection of M_immediate onto \nu
        M_projection = (self.papw.T * self.nu).T

        #Calculate scaling factors
        B_norm = norm(self.B_forwards)
        A_norm = norm(self.A)
        M_norm = norm(M_projection)
        self.p0 = np.sqrt(B_norm/A_norm)
        self.p1 = np.sqrt(M_norm/np.sqrt(self.n_h))

        #Override with fixed P0 and P1 if given
        if self.P0 is not None:
            self.p0 = np.copy(self.P0)
        if self.P1 is not None:
            self.p1 = np.copy(self.P1)

        #Update "inverse" Kronecker product approximation
        A = self.p0 * self.A + self.p1 * self.nu
        B = (1/self.p0) * self.B_forwards + (1/self.p1) * M_projection

        return A, B

    def get_rec_grads(self):
        """Calculates recurrent grads by taking matrix product of q with the
        estimate of the influence matrix.

        First associates q with B to calculate qB, then takes the outer product
        with A to get an estimate of the recurrent gradient.

        Returns:
            An array of shape (n_h, m) representing the recurrent gradient."""

        self.qB = self.q.dot(self.B) #Unit-specific learning signal
        return np.multiply.outer(self.A, self.qB)

    def reset_learning(self):
        """Resets learning by re-randomizing the Kron. product approximation to
        random gaussian samples."""

        self.A = np.random.normal(0, 1, self.n_h)
        self.B = np.random.normal(0, 1/np.sqrt(self.n_h),
                                  (self.n_h, self.m))

class RFLO(Real_Time_Learning_Algorithm):
    """Implements the Random-Feedback Local Online learning algorithm (RFLO)
    from Murray (2019).

    Maintains an eligibility trace B that is updated by temporally filtering
    the immediate influences \phi'(h_i) a_hat_j by the network's inverse time
    constant \alpha:

    B'_{ij} = (1 - \alpha) B_{ij} + \alpha \phi'(h_i) a_hat_j       (1)

    Eq. (1) is implemented by update_learning_vars method. Gradients are then
    calculated according to

    q_i B_{ij}      (2)

    which is implemented in get_rec_grads."""


    def __init__(self, net, alpha, **kwargs):
        """Inits an RFLO instance by specifying the inverse time constant for
        the eligibility trace.

        Args:
            alpha (float): Float between 0 and 1 specifying the inverse time
                constant of the eligilibility trace, typically chosen to be
                equal to alpha for the network.

        Keyword args:
            B (numpy array): Initial value for B (all 0s if unspecified)."""

        self.name = 'RFLO'
        allowed_kwargs_ = {'B'}
        super().__init__(net, allowed_kwargs_, **kwargs)

        self.alpha = alpha
        if self.B is None:
            self.B = np.zeros((self.n_h, self.m))

    def update_learning_vars(self):
        """Updates B by one time step of temporal filtration via the invesre
        time constant alpha (see Eq. 1)."""

        #Get relevant values and derivatives from network
        self.a_hat = np.concatenate([self.net.a_prev,
                                     self.net.x,
                                     np.array([1])])
        self.D = self.net.activation.f_prime(self.net.h)
        self.M_immediate = self.alpha * np.multiply.outer(self.D,
                                                          self.a_hat)

        #Update eligibility traces
        self.B = (1 - self.alpha) * self.B + self.M_immediate

    def get_rec_grads(self):
        """Implements Eq. (2) from above."""

        return (self.q * self.B.T).T

    def reset_learning(self):
        """Reset eligibility trace to 0."""

        self.B *= 0

class DNI(Real_Time_Learning_Algorithm):
    """Implements the Decoupled Neural Interface (DNI) algorithm for an RNN.

    Details are in Jaderberg et al. (2017). Briefly, we linearly approximate
    the (future-facing) credit assignment vector c = dL/da using


    """
    def __init__(self, net, optimizer, **kwargs):

        self.name = 'DNI'
        allowed_kwargs_ = {'SG_clipnorm', 'SG_target_clipnorm', 'J_lr',
                           'activation', 'SG_label_activation', 'use_approx_J',
                           'SG_L2_reg', 'fix_SG_interval'}
        #Default parameters
        self.optimizer = optimizer
        self.SG_L2_reg = 0
        self.fix_SG_interval = 5
        self.activation = identity
        self.SG_label_activation = identity
        self.use_approx_J = False
        #Override defaults with kwargs
        super().__init__(net, allowed_kwargs_, **kwargs)

        sigma = np.sqrt(1/self.n_h)
        self.m_out = self.n_h + self.n_out + 1
        self.SG_init(sigma)
        self.J_approx = np.copy(self.net.W_rec)
        self.i_fix = 0
        self.A_= np.copy(self.A)

    def SG_init(self, sigma):

        self.A = np.random.normal(0, sigma, (self.n_h, self.m_out))

    def update_learning_vars(self):


        #Get network jacobian
        self.net.get_a_jacobian()

        #Compute synthetic gradient estimate of credit assignment
        self.a_tilde_prev = np.concatenate([self.net.a_prev,
                                            self.net.y_prev,
                                            np.array([1])])
        self.sg = self.synthetic_grad(self.a_tilde_prev)

        if self.SG_clipnorm is not None:
            self.sg_norm = norm(self.sg)
            if self.sg_norm > self.SG_clipnorm:
                self.sg = self.sg / self.sg_norm

        self.sg_target = self.get_sg_target()

        if self.SG_target_clipnorm is not None:
            self.sg_target_norm = norm(self.sg_target)
            if self.sg_target_norm > self.SG_target_clipnorm:
                self.sg_target = self.sg_target / self.sg_target_norm

        self.e_sg = self.sg - self.sg_target
        self.sg_loss = np.mean((self.sg - self.sg_target)**2)
        self.scaled_e_sg = self.e_sg*self.activation.f_prime(self.sg_h)

        #Get SG grads
        #self.SG_grads = [np.multiply.outer(self.scaled_e_sg, self.net.a_prev),
        #                 np.multiply.outer(self.scaled_e_sg, self.net.y_prev),
        #                 self.scaled_e_sg]

        self.SG_grads = np.multiply.outer(self.scaled_e_sg, self.a_tilde_prev)

        if self.SG_L2_reg > 0:
            self.SG_grad += self.SG_L2_reg*self.A

        #Update SG parameters
        self.A = self.optimizer.get_updated_params([self.A], [self.SG_grads])[0]

        if self.i_fix == self.fix_SG_interval - 1:
            self.i_fix = 0
            self.A_ = np.copy(self.A)
        else:
            self.i_fix += 1

        if self.J_lr is not None:
            self.update_J_approx()

    def get_sg_target(self):

        self.propagate_feedback_to_hidden()

        self.a_tilde = np.concatenate([self.net.a, self.net.y, np.array([1])])

        if self.use_approx_J:
            sg_target = self.q_prev + self.synthetic_grad_(self.a_tilde).dot(self.J_approx)
        else:
            sg_target = self.q_prev + self.synthetic_grad_(self.a_tilde).dot(self.net.a_J)

        return sg_target

    def update_J_approx(self):

        self.loss_a = np.square(self.J_approx.dot(self.net.a_prev) - self.net.a).mean()
        self.e_a = self.J_approx.dot(self.net.a_prev) - self.net.a

        self.J_approx -= self.J_lr*np.multiply.outer(self.e_a, self.net.a_prev)

    def synthetic_grad(self, a_tilde):
        #self.sg_h = self.A.dot(a) + self.B.dot(y) + self.C
        self.sg_h = self.A.dot(a_tilde)
        return self.activation.f(self.sg_h)

    def synthetic_grad_(self, a_tilde):
        self.sg_h_ = self.A_.dot(a_tilde)
        return self.SG_label_activation.f((self.activation.f(self.sg_h_)))

    def get_rec_grads(self):

        #self.sg = self.synthetic_grad(self.net.a, self.net.y)
        self.sg = self.synthetic_grad(self.a_tilde)
        self.sg_scaled = self.net.alpha*self.sg*self.net.activation.f_prime(self.net.h)

        if self.SG_clipnorm is not None:
            sg_norm = norm(self.sg)
            if sg_norm > self.SG_clipnorm:
                self.sg = self.sg / sg_norm

        self.a_hat = np.concatenate([self.net.a_prev, self.net.x, np.array([1])])

        return np.multiply.outer(self.sg_scaled, self.a_hat)

class BPTT(Learning_Algorithm):

    def __init__(self, net, t1, t2, monitors=[], use_historical_W=False):

        super().__init__(net, monitors)

        #The two integer parameters
        self.t1  = t1    #Number of steps to average loss over
        self.t2  = t2    #Number of steps to propagate backwards
        self.T   = t1+t2 #Total amount of net hist needed to memorize
        self.use_historical_W = use_historical_W

        #Lists storing relevant net history
        self.h_hist = [np.zeros(self.net.n_h) for _ in range(self.T)]
        self.a_hist = [np.zeros(self.net.n_h) for _ in range(self.T)]
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

        n_h, n_in = self.net.n_h, self.net.n_in
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

class BPTT_Triangles(Real_Time_Learning_Algorithm):

    def __init__(self, net, T_truncation, **kwargs):

        self.name = 'BPTT_Triangles'
        allowed_kwargs_ = set()
        super().__init__(net, allowed_kwargs_, **kwargs)

        self.T_truncation = T_truncation

        self.CA_hist = [np.zeros(self.n_h)]*T_truncation
        self.a_hat_hist = [np.zeros(self.m)]*T_truncation
        self.h_hist = [np.zeros(self.n_h)]*T_truncation
        self.q_hist = [np.zeros(self.n_h)]*T_truncation

        self.i_t = 0

    def update_learning_vars(self):


        #Update history
        self.a_hat = np.concatenate([self.net.a_prev, self.net.x, np.array([1])])
        self.propagate_feedback_to_hidden()

        self.a_hat_hist[self.i_t] = np.copy(self.a_hat)
        self.h_hist[self.i_t] = np.copy(self.net.h)
        self.q_hist[self.i_t] = np.copy(self.q)

        self.i_t += 1
        if self.i_t == self.T_truncation:
            self.i_t = 0



        for i_CA in range(len(self.CA_hist)):

            q = np.copy(self.q)
            #J = np.eye(self.n_h)

            for i_BP in range(i_CA):

                J = self.net.get_a_jacobian(update=False,
                                            h=self.h_hist[-(i_BP+1)])
                q = q.dot(J)
                #J = J.dot(self.net.get_a_jacobian(update=False, h=self.h_hist[-(i_BP+1)]))

            #self.CA_hist[-(i_CA + 1)] += self.q.dot(J)
            self.CA_hist[-(i_CA + 1)] += q

    def get_rec_grads(self):

        if self.i_t > 0:

            rec_grads = np.zeros((self.n_h, self.n_h + self.n_in + 1))

        else:

            pass

        if len(self.CA_hist)==self.T_truncation:

            self.net.CA = np.copy(self.CA_hist[0])

            self.D = self.net.activation.f_prime(self.h_hist[0])
            rec_grads = np.multiply.outer(self.net.CA*self.D, self.a_hat_hist[0])

            self.delete_history()

        else:

            pass

        return rec_grads

    def reset_learning(self):

        pass

class Forward_BPTT(Real_Time_Learning_Algorithm):

    def __init__(self, net, T_truncation, **kwargs):

        self.name = 'F-BPTT'
        allowed_kwargs_ = set()
        super().__init__(net, allowed_kwargs_, **kwargs)

        self.T_truncation = T_truncation

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

#        for i_CA in range(len(self.CA_hist)):
#
#            q = np.copy(self.q)
#            #J = np.eye(self.n_h)
#
#            for i_BP in range(i_CA):
#
#                J = self.net.get_a_jacobian(update=False,
#                                            h=self.h_hist[-(i_BP+1)])
#                q = q.dot(J)
#                #J = J.dot(self.net.get_a_jacobian(update=False, h=self.h_hist[-(i_BP+1)]))
#
#            #self.CA_hist[-(i_CA + 1)] += self.q.dot(J)
#            self.CA_hist[-(i_CA + 1)] += q

        q = np.copy(self.q)

        for i_BP in range(len(self.CA_hist)):

            self.CA_hist[-(i_BP + 1)] += q
            J = self.net.get_a_jacobian(update=False,
                                        h=self.h_hist[-(i_BP+1)])
            q = q.dot(J)
            #J = J.dot(self.net.get_a_jacobian(update=False, h=self.h_hist[-(i_BP+1)]))

        #self.CA_hist[-(i_CA + 1)] += self.q.dot(J)


    def get_rec_grads(self):

        if len(self.CA_hist)==self.T_truncation:

            self.net.CA = np.copy(self.CA_hist[0])

            self.D = self.net.activation.f_prime(self.h_hist[0])
            rec_grads = np.multiply.outer(self.net.CA*self.D, self.a_hat_hist[0])

            self.delete_history()

        else:

            rec_grads = np.zeros((self.n_h, self.n_h + self.n_in + 1))

        return rec_grads

    def reset_learning(self):

        self.CA_hist = []
        self.a_hat_hist = []
        self.h_hist = []

    def delete_history(self):

        for attr in ['CA', 'a_hat', 'h']:
            del(self.__dict__[attr+'_hist'][0])

class KeRNL(Real_Time_Learning_Algorithm):

    def __init__(self, net, optimizer, sigma_noise=0.00001,
                 use_approx_kernel=False, learned_alpha_e=False,
                 **kwargs):

        self.name = 'KeRNL'
        self.n_h = net.n_h
        self.n_in = net.n_in
        self.i_t = 0
        self.sigma_noise = sigma_noise
        self.optimizer = optimizer
        self.use_approx_kernel = use_approx_kernel
        self.learned_alpha_e = learned_alpha_e
        self.zeta = np.random.normal(0, self.sigma_noise, self.n_h)

        #Initialize learning variables
        #self.beta = np.random.normal(0, 1/np.sqrt(self.n_h), (self.n_h, self.n_h))
        self.beta = np.eye(self.n_h)
        #self.gamma = (1/10)**np.random.uniform(0, 2, self.n_h)
        if use_approx_kernel:
            self.gamma = np.ones(self.n_h)*0.8
        else:
            self.gamma = (1/10)**np.random.uniform(0, 2, self.n_h)
        self.eligibility = np.zeros((self.n_h, self.n_h + self.n_in + 1))
        self.Omega = np.zeros(self.n_h)
        self.Gamma = np.zeros(self.n_h)

        #Initialize noisy network
        self.noisy_net = copy(net)

        allowed_kwargs_ = {'beta', 'gamma', 'Omega',
                           'Gamma', 'eligibility', 'T_reset'}
        super().__init__(net, allowed_kwargs_, **kwargs)

        if use_approx_kernel:
            self.kernel = self.approx_kernel
        else:
            self.kernel = self.exact_kernel

    def exact_kernel(self, delta_t):

        return np.maximum(0, np.exp(-self.gamma*(delta_t)))

    def approx_kernel(self, delta_t):

        return np.maximum(0, 1 - self.gamma*delta_t)

    def update_learning_vars(self):

        #if self.i_t > 0 and self.i_t%1000 == 0 and False:
        #    set_trace()

        #Observe Jacobian if desired:
        self.J = self.net.get_a_jacobian(update=False)

        #Update noisy net's parameters
        self.noisy_net.W_rec = self.net.W_rec
        self.noisy_net.W_in = self.net.W_in
        self.noisy_net.b_rec = self.net.b_rec

        #Update noisy net forward
        if self.T_reset is not None:
            if self.i_t % self.T_reset == 0:
                self.reset_learning()

        self.noisy_net.a += self.zeta
        self.noisy_net.next_state(self.net.x)

        #Update learning varialbes
        self.zeta = np.random.normal(0, self.sigma_noise, self.n_h)
        if self.use_approx_kernel:
            self.Gamma = self.kernel(1)*self.Gamma - self.Omega
        else:
            self.Gamma = self.kernel(1)*(self.Gamma - self.Omega)
        self.Omega = self.kernel(1)*self.Omega + (1 - self.kernel(1))*self.zeta

        #Update eligibility traces
        self.D = self.net.activation.f_prime(self.net.h)
        self.a_hat = np.concatenate([self.net.a_prev, self.net.x, np.array([1])])
        if self.learned_alpha_e:
            self.papw = np.multiply.outer((1 - self.kernel(1))*self.D, self.a_hat)
        else:
            self.papw = self.net.alpha*np.multiply.outer(self.D, self.a_hat)
        self.eligibility = (self.eligibility.T*self.kernel(1)).T + self.papw

        #Get error in predicting perturbations effect
        self.error_prediction = self.beta.dot(self.Omega)
        self.error_observed = (self.noisy_net.a - self.net.a)
        self.loss_noise = np.square(self.error_prediction - self.error_observed).sum()
        #self.loss_noise = np.square((self.beta.dot(self.Omega) - (self.noisy_net.a - self.net.a))).sum()
        self.e_noise = self.error_prediction - self.error_observed
        #self.e_noise = self.beta.dot(self.Omega) - (self.noisy_net.a - self.net.a)

        #Update beta and gamma
        self.beta_grads = np.multiply.outer(self.e_noise, self.Omega)
        self.gamma_grads = self.e_noise.dot(self.beta)*self.Gamma
        self.beta, self.gamma = self.optimizer.get_updated_params([self.beta, self.gamma],
                                                                  [self.beta_grads, self.gamma_grads])

        self.i_t += 1

    def get_rec_grads(self):

        return (self.eligibility.T*self.q.dot(self.beta)).T

    def reset_learning(self):

        self.noisy_net.a = np.copy(self.net.a)
        self.Omega = np.zeros_like(self.Omega)
        self.Gamma = np.zeros_like(self.Gamma)
        self.eligibility = np.zeros_like(self.eligibility)

class Random_Walk_RTRL(Real_Time_Learning_Algorithm):
    """Algorithm idea combining tensor structure of KeRNL with stochastic
    update principles of KF-RTRL."""

    def __init__(self, net, rho_A=1, rho_B=1, gamma=0.5, **kwargs):

        self.name = 'RW-RTRL'
        allowed_kwargs_ = set()
        super().__init__(net, allowed_kwargs_, **kwargs)

        self.gamma = gamma
        self.rho_A = rho_A
        self.rho_B = rho_B
        self.A = np.zeros((self.n_h, self.m))
        self.B = np.zeros((self.n_h, self.n_h))

    def update_learning_vars(self):

        self.net.get_a_jacobian()
        a_J = self.net.a_J

        self.nu = np.random.choice([-1, 1], self.n_h)

        self.a_hat = np.concatenate([self.net.a_prev,
                                     self.net.x,
                                     np.array([1])])
        self.D = self.net.activation.f_prime(self.net.h)

        self.e_ij = np.multiply.outer(self.D**(1-self.gamma),
                                      self.a_hat)
        self.e_ki = np.diag(self.net.alpha * self.D**self.gamma)

        self.A = self.A + (self.nu * (self.rho_A * self.e_ij).T).T
        self.B = a_J.dot(self.B) + self.nu*self.rho_B*self.e_ki

    def get_rec_grads(self):

        return (self.q.dot(self.B) * self.A.T).T

class Reward_Modulated_Hebbian_Plasticity(Real_Time_Learning_Algorithm):
    """Implements a reward-modulated Hebbian plasticity rule for *trial-
    structured* tasks only (for now)."""

    def __init__(self, net, alpha, task, **kwargs):

        self.name = 'RM-Hebb'
        allowed_kwargs_ = {'B', 'fixed_modulation'}
        super().__init__(net, allowed_kwargs_, **kwargs)

        self.alpha = alpha
        if self.B is None:
            self.B = np.zeros((self.n_h, self.m))

        self.task = task

        self.running_loss_avg = 0
        self.i_t = 0

    def update_learning_vars(self):
        """Updates B by one time step of temporal filtration via the invesre
        time constant alpha (see RFLO)."""

        self.i_trial = self.i_t % self.task.time_steps_per_trial

        #Get relevant values and derivatives from network
        self.a_hat   = np.concatenate([self.net.a_prev,
                                       self.net.x,
                                       np.array([1])])
        self.D = self.net.activation.f_prime(self.net.h)
        self.M_immediate = self.alpha * np.multiply.outer(self.D,
                                                          self.a_hat)

        #Update eligibility traces
        self.B = (1 - self.alpha) * self.B + self.M_immediate
        #self.B = (1 - self.alpha) * self.B + self.alpha * np.multiply.outer(self.net.a,
        #                                                                    self.a_hat)

        if self.task.trial_lr_mask[self.i_trial] > 0.3:
            #Update running loss average
            self.running_loss_avg = 0.99 * self.running_loss_avg + 0.01 * self.net.loss_

        self.i_t += 1

    def get_rec_grads(self):
        """Scales Hebbian plasticity rule by loss running average."""

        if self.task.trial_lr_mask[self.i_trial] > 0.3:
            scale = 1
        else:
            scale = 0

        if self.fixed_modulation is not None:
            self.modulation = self.fixed_modulation
        else:
            self.modulation = scale * (self.net.loss_ - self.running_loss_avg)
        return self.modulation * self.B

    def reset_learning(self):
        """Reset eligibility trace to 0."""

        self.B *= 0

class COLIN(Real_Time_Learning_Algorithm):
    """Implements Cholinergic Operant Learning In Neurons"""

    def __init__(self, net, decay, sigma, task, **kwargs):

        self.name = 'COLIN'
        allowed_kwargs_ = {'B', 'fixed_modulation'}
        super().__init__(net, allowed_kwargs_, **kwargs)

        self.decay = decay
        if self.B is None:
            self.B = np.zeros((self.n_h, self.m))

        self.task = task
        self.sigma = sigma
        self.alpha_loss = 0.99
        self.running_loss_avg = 0
        self.i_t = 0

    def update_learning_vars(self):
        """Updates B by one time step of temporal filtration via the invesre
        time constant alpha (see RFLO)."""

        self.i_trial = self.i_t % self.task.time_steps_per_trial

        #Get relevant values and derivatives from network
        self.a_hat   = np.concatenate([self.net.a_prev,
                                       self.net.x,
                                       np.array([1])])
        self.D = self.net.activation.f_prime(self.net.h)
        self.D_noise = self.D * (self.net.noise/self.sigma**2)
        #self.D_noise = self.D *(1/self.sigma**2)
        #self.D_noise = self.D * (self.sigma*self.net.alpha/self.sigma**2)
        self.M_immediate = self.net.alpha * np.multiply.outer(self.D_noise,
                                                              self.a_hat)

        #Update eligibility traces
        #np.multiply.outer(self.net.noise/self.net.sigma, self.net.a_prev)
        self.B = (1 - self.decay) * self.B + self.M_immediate
        #self.B = (1 - self.alpha) * self.B + self.alpha * np.multiply.outer(self.net.a,
        #                                                                    self.a_hat)

        if self.task.trial_lr_mask[self.i_trial] > 0.3:
            #Update running loss average
            self.running_loss_avg = ((1 - self.alpha_loss) * self.running_loss_avg +
                                     self.alpha_loss * self.net.loss_)

        self.i_t += 1

    def get_rec_grads(self):
        """Scales Hebbian plasticity rule by loss running average."""

        if self.task.trial_lr_mask[self.i_trial] > 0.3:
            scale = 1
        else:
            scale = 0

        if self.fixed_modulation is not None:
            self.modulation = self.fixed_modulation
        else:
            self.modulation = scale * (self.net.loss_ - self.running_loss_avg)
        return self.modulation * self.B

    def reset_learning(self):
        """Reset eligibility trace to 0."""

        self.B *= 0









































