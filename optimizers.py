#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

class Optimizer:
    """Parent class for gradient-based optimizers."""

    def __init__(self, allowed_kwargs_, **kwargs):

        allowed_kwargs = {'lr_decay_rate', 'min_lr',
                          'clip_norm'}.union(allowed_kwargs_)
        for k in kwargs:
            if k not in allowed_kwargs:
                raise TypeError('Unexpected keyword argument '
                                'passed to optimizer: ' + str(k))
        self.__dict__.update(kwargs)

    def clip_gradient(self, grads):
        """Clips each gradient by the global gradient norm if it exceeds
        self.clip_norm.

        Throws error if self.clip_norm attribute does not exist.

        Args:
            grads (list): List of original gradients
        Returns:
            clipped_grads (list): List of clipped gradients."""

        
        clipped_grads = []
        for grad in grads:
            clipped_grads.append(grad*self.clip_norm/)

    def lr_decay(self):

        self.lr_ = self.lr_*self.lr_decay_rate
        try:
            return np.max([self.lr_, self.min_lr])
        except AttributeError:
            return self.lr_

class SGD(Optimizer):
    """Implements basic stochastic gradient descent optimizer.

    Attributes:
        lr (float): learning rate. """

    def __init__(self, lr=0.001, **kwargs):

        allowed_kwargs_ = set()
        super().__init__(allowed_kwargs_, **kwargs)

        self.lr_ = np.copy(lr)
        self.lr = lr

    def get_updated_params(self, params, grads):
        """Returns a list of updated parameter values (NOT the change in value).

        Args:
            params (list): List of trainable parameters as numpy arrays
            grads (list): List of corresponding gradients as numpy arrays.
        Returns:
            updated_params (list): List of newly updated parameters."""

        if hasattr(self, 'lr_decay_rate'):
            self.lr = self.lr_decay()

        updated_params = []
        for param, grad in zip(params, grads):
            updated_params.append(param - self.lr * grad)

        return updated_params

