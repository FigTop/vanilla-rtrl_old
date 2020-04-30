#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from utils import *
from functions import *


class LSTM:
    """ Long short-term memory network.
    Forward equation
    f_t = sigma (W_f h_hat_{t-1} + b_f)
    i_t = sigma (W_i h_hat_{t-1} + b_i)
    a_t = tanh (W_a h_hat_{t-1} + b_a)
    c_t = a_t * i_t +  f_t * c_{t-1}
    o_t = sigma(W_o h_hat_{t-1} + b_o)
    h_t = tanh(c_t) * o_t
    z_t = W_h_out h_t + W_c_out c_t + b_out

    Attributes:
        n_in (int): Number of input dimensions
        n_h (int): Number of hidden units or cell units
        n_t (int): Total number of hidden and cell units =2n_h
        n_h_hat (int): Number of hidden units + number of input dimensions
        n_out (int): Number of output dimensions
        W_f (numpy array): Array of shape (n_h, n_h_hat), weights forget gate inputs.
        W_i (numpy array): Array of shape (n_h, n_h_hat), weights input gate inputs.
        W_a (numpy array): Array of shape (n_h, n_h_hat), weights activation inputs.
        W_o (numpy array): Array of shape (n_h, n_h_hat), weights output gate inputs.
        W_out (numpy array): Array of shape (n_out, n_t), provides [hidden,cell]-to-
            output-layer weights.
        b_f (numpy array): Array of shape (n_h), represents the bias term in the forget gate update equation.
        b_i (numpy array): Array of shape (n_h), represents the bias term in the input gate update equation.
        b_a (numpy array): Array of shape (n_h), represents the bias term in the activation update equation.
        b_o (numpy array): Array of shape (n_h), represents the bias term in the output gate update equation.
        b_out (numpy array): Array of shape (n_h), represents the bias term in the final output update equation.

        params (list): The list of each parameter's current value, in the order
            [self.W_f, self.W_i, self.W_a, self.W_o, self.b_f, self.b_i, self.b_a, self.b_o,
            self.W_out, self.b_out]
        shapes (list): The shape of each trainable set of parameters, in the
            same order.
        n_params (int): Number of total trainable parameters.
        n_h_params (int): Number of total trainble parameters in the recurrent update equation, i.e. all parameters excluding the output weights and biases.
        L2_indices (list): A list of integers representing which indices in
            the list of params should be subject to L2 regularization if the
            learning algorithm dictates (all weights but not biases).

        sigmoid (functions.Function): An instance of the Function class used as the network's nonlinearity.
        tanh (functions.Function): An instance of the Function class used as the network's nonlinearity.
        output (functions.Function): An instance of the Function class used
            for calculating final output from z.
        loss (functions.Function): An instance of the Function class used for
            calculating loss from z (must implicitly include output function,
            e.g. softmax_cross_entropy if output is softmax).

        x (numpy array): Array of shape (n_in) representing the current inputs
            to the network.
        h (numpy array): Array of shape (n_h) representing the hidden state of the network.
        h_hat (numpy array): Array of shape (n_h_hat) representing the cat(hidden state, inputs x) of the network.
        c (numpy array): Array of shape (n_h) representing the cell state of the network.
        f (numpy array): Array of shape (n_h) representing the forget gate of the network.
        i (numpy array): Array of shape (n_h) representing the input gate of the network.
        a (numpy array): Array of shape (n_h) representing the activation of the network.
        o (numpy array): Array of shape (n_h) representing the output gate of the network.
        z (numpy array): Array of shape (n_out) reprenting the outputs of the
            network, before any final output nonlinearities, e.g. softmax,
            are applied.
        *_prev (numpy array): Array representing any of x, h, c, or z at the
            previous time step.

        error (numpy array): Array of shape (n_out) representing the derivative
            of the loss with respect to z. Calculated by loss.f_prime.
        y_hat (numpy array): Array of shape (n_out) representing the final
            outputs of the network, to be directly compared with task labels.
            Not computed in any methods in this class.

        """


    def __init__(self, W_f, W_i, W_a, W_o, W_out,
                 b_f, b_i, b_a, b_o, b_out,
                 output, loss,
                 sigmoid = Function(sigmoid_,sigmoid_derivative),
                 tanh = Function(tanh_,tanh_derivative)):
        """Initializes a LSTM by specifying its initial parameter values;
        its activation(sigmoid and tanh), output, and loss functions;."""

        #Initial parameter values
        self.W_out = W_out
        self.b_out = b_out

        self.W_f = W_f
        self.W_i = W_i
        self.W_a = W_a
        self.W_o = W_o
        self.b_f = b_f
        self.b_i = b_i
        self.b_a = b_a
        self.b_o = b_o

        # Network dimensions
        self.n_h_hat = W_f.shape[1]
        self.n_h = W_f.shape[0]
        self.n_t = 2 * self.n_h
        self.n_in = W_f.shape[1] - W_f.shape[0]
        self.n_out = W_out.shape[0]
        self.m = (self.n_h_hat+1)*4

        #Check dimension consistency.
        assert self.n_h == W_a.shape[0]
        assert self.n_h_hat == W_a.shape[1]
        assert self.n_h == W_i.shape[0]
        assert self.n_h_hat == W_i.shape[1]
        assert self.n_h == W_o.shape[0]
        assert self.n_h_hat == W_o.shape[1]

        assert self.n_h == b_f.shape[0]
        assert self.n_h == b_i.shape[0]
        assert self.n_h == b_a.shape[0]
        assert self.n_h == b_o.shape[0]
        assert self.n_out == W_out.shape[0]
        assert self.n_out == b_out.shape[0]


        #Define shapes and params lists for convenience later.
        self.params = [self.W_f, self.b_f,
                       self.W_i, self.b_i,
                       self.W_a, self.b_a,
                       self.W_o, self.b_o,
                       self.W_out, self.b_out]
        self.shapes = [w.shape for w in self.params]

        self.param_names = ['W_f', 'b_f',
                            'W_i', 'b_i',
                            'W_a', 'b_a',
                            'W_o', 'b_o',
                            'W_out', 'b_out']

        #Activation and loss functions
        self.sigmoid = sigmoid
        self.tanh = tanh
        self.loss = loss
        self.output = output

        #Number of parameters
        self.n_h_params = (self.W_f.size + self.b_f.size )*4

        self.n_params = (self.n_h_params +
                         self.W_out.size + self.b_out.size)

        #Indices of params for L2 regularization
        self.L2_indices = [0, 2, 4, 6, 8, 9] # W_f, W_i, W_a, W_o, W_out

        #Initialize influence matrix
        self.papwf = np.zeros((self.n_t, int(self.n_h_params/4)))
        self.papwi = np.zeros((self.n_t, int(self.n_h_params/4)))
        self.papwa = np.zeros((self.n_t, int(self.n_h_params/4)))
        self.papwo = np.zeros((self.n_t, int(self.n_h_params/4)))


        #Initial state values
        self.reset_network()

    def reset_network(self, sigma=1, **kwargs):
        """Resets hidden state and cell state of the network, either randomly or by
        specifying with kwargs.
        Args:
            sigma (float): Standard deviation of (zero-mean) Gaussian random
                reset of pre-activations h. Used if neither h or a is
                specified.
            h (numpy array): The specification of the hidden state values,
                must be of shape (self.n_h). The a values are determined by
                application of the nonlinearity to h.
            c (numpy array): The specification of the cell state values,
                must be of shape (self.n_h). If not specified, determined
                by h."""

        if 'h' in kwargs.keys(): #Manual reset if specified.
            self.h = kwargs['h']
        else: #Random reset by sigma if not.
            self.h = np.random.normal(0, sigma, self.n_h)

        if 'c' in kwargs.keys():
            self.c = kwargs['c']
        else:
            self.c = np.random.normal(0, sigma, self.n_h)

        self.state = np.concatenate([self.c,self.h])

        self.prev_h = self.h
        self.prev_c = self.c
        self.z = self.W_out.dot(self.state)+ self.b_out #Specify outputs from a


    def next_state(self, x, c=None, h= None, update=True, sigma=0):
        """Advances the network forward by one time step.
        Accepts as argument the current time step's input x and updates
        the state of the LSTM, while storing the previous hidden state h
        and cell state c. Can either update the network (if update=True)
        or return what the update would be.
        Args:
            x (numpy array): Input provided to the network, of shape (n_in).
            update (bool): Specifies whether to update the network using the
                current network state (if True) or return the would-be next
                network state using a provided "current" network state a.
            h (numpy array): Recurrent inputs used to drive the network, to be
                provided only if update is False.
            c (numpy array): cell state inputs used to drive the network, to be
                provided only if update is False.
            sigma (float): Standard deviation of white noise added to pre-
                activations before applying \phi.
        Returns:
            Updates self.x, self.h,self.c, and self.*_prev, or returns the
            would-be update from given previous state h and c."""

        if update:
            self.h_prev = np.copy(self.h)
            self.x = x
            self.h_hat_prev = np.append(self.h_prev, self.x, axis=0)
            self.c_prev = np.copy(self.c)

            self.f = self.sigmoid.f(self.W_f.dot(self.h_hat_prev)+self.b_f)
            self.i = self.sigmoid.f(self.W_i.dot(self.h_hat_prev)+self.b_i)
            self.a = self.tanh.f(self.W_a.dot(self.h_hat_prev)+self.b_a)

            self.c = self.a * self.i + self.f * self.c_prev
            self.o = self.sigmoid.f(self.W_f.dot(self.h_hat_prev)+self.b_o)
            self.h = self.tanh.f(self.c) * self.o
            self.state = np.concatenate([self.c,self.h])

        else:
            h_hat_prev = np.append(h, self.x, axis=0)

            f = self.sigmoid.f(self.W_f.dot(h_hat_prev)+self.b_f)
            i = self.sigmoid.f(self.W_i.dot(h_hat_prev)+self.b_i)
            a = self.tanh.f(self.W_a.dot(h_hat_prev)+self.b_a)

            c = a * i + f * c
            o = self.sigmoid.f(self.W_f.dot(h_hat_prev)+self.b_o)
            h = self.tanh.f(c) * o

            return np.concatenate([c, h])


    def z_out(self):
        """Update outputs using current state of the network."""

        self.z_prev = np.copy(self.z)
        self.z = self.W_out.dot(self.state)+ self.b_out


    def get_a_jacobian(self, update=True, x=None, c=None, h=None):
        """Calculates the Jacobian of the network.
        J(c_k/c_i) = f_k^{t}
        J(h_k/c_i) = o_k^{t} tanh'(c_k^{t}) * J(c_k/c_i)

        J(c_k/h_i) = w_{ki}^{a} i_k^{t} tanh'(a_k^{t}) +
                     w_{ki}^{i} a_k^{(t)} sigmoid'(i_k^{t}) +
                     w_{ki}^{f} c_k^{t-1} sigmoid'(f_k^{t})

        J(h_k/h_i) = o_k^{t} tanh'(c_k^{t})J(c_k/h_i) +
                    tanh(c_k^{t}) w^o_{ki} sigmoid'(o_k^{t})
        Args:
            update (bool): Specifies whether to update or return the Jacobian.
            h (numpy array): Array of shape (n_h) that specifies what values of
                the hidden states to use in calculating the Jacobian.
            c (numpy array): Array of shape (n_h) that specifies what values of
                the cell states to use in calculating the Jacobian.
            x (numpy array): Array of shape (n_in) that specifies what values of
                the input value to use in calculating the Jacobian."""


        #Use kwargs instead of defaults if provided
        if x == None:
            x = self.x
        if c == None:
            c = self.c
        if h == None:
            h = self.h


        h_hat_prev = np.append(h, x, axis=0)
        f = self.sigmoid.f(self.W_f.dot(h_hat_prev) + self.b_f)
        i = self.sigmoid.f(self.W_i.dot(h_hat_prev) + self.b_i)
        a = self.tanh.f(self.W_a.dot(h_hat_prev) + self.b_a)

        c_prev = c
        c = a * i + f * c_prev
        o = self.sigmoid.f(self.W_f.dot(h_hat_prev) + self.b_o)
        h = self.tanh.f(c) * o


        # Calculate four parts of Jacobian
        c_c_J = np.eye(self.n_h) * f

        h_c_J = (c_c_J.T * (o * self.tanh.f_prime(c))).T

        P_1 = i * (1-a**2)
        P_2 = a * (i-i**2)
        P_3 = c_prev * (f-f**2)
        c_h_J = (self.W_a[:,:self.n_h].T * P_1 +
                 self.W_i[:,:self.n_h].T * P_2 +
                 self.W_f[:,:self.n_h].T * P_3).T


        P_4 = self.tanh.f(c)* (o-o**2)
        h_h_J = (c_h_J.T * (o* self.tanh.f_prime(c)) +
               (self.W_o[:,:self.n_h].T * P_4)).T

        # stack four parts together to get entire Jacobian
        a_J = np.vstack((np.hstack((c_c_J,c_h_J)),
                        np.hstack((h_c_J,h_h_J))))

        if update: #Update if update is True
            self.a_J = np.copy(a_J)
        else: #Otherwise return
            return a_J

    def update_M_immediate(self):
        """Updates the influence matrix via Eq. (1)."""

        self.state_hat = np.concatenate([self.h_hat_prev, np.array([1])])
        
        #Calculate M_immediate
        
        r = self.o * self.tanh.f_prime(self.c)

        # self.m = n_h_prev+1
        pcpwo = np.zeros((self.n_h, (self.n_h_hat+1) * self.n_h))
        D_o = np.diag((self.o-self.o**2) * self.tanh.f(self.c))
        phpwo = np.kron(self.state_hat, D_o)
        self.papwo = np.concatenate([pcpwo, phpwo])

        D_a = np.diag((1-self.a**2) * self.i)
        pcpwa = np.kron(self.state_hat, D_a)
        phpwa = (pcpwa.T * r).T
        self.papwa = np.concatenate([pcpwa, phpwa])

        D_i = np.diag((self.i-self.i**2) * self.a)
        pcpwi = np.kron(self.state_hat, D_i)
        phpwi = (pcpwi.T * r).T
        self.papwi = np.concatenate([pcpwi, phpwi])

        D_f = np.diag((self.f-self.f**2) * self.c_prev)
        pcpwf = np.kron(self.state_hat, D_f)
        phpwf = (pcpwf.T * r).T
        self.papwf = np.concatenate([pcpwf, phpwf])

        self.papw = np.concatenate([self.papwf,self.papwi,self.papwa,self.papwo],axis=1)