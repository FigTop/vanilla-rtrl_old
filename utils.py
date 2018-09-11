#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 14:25:20 2018

@author: omarschall
"""

import numpy as np

class function():
    
    def __init__(self, f, f_prime):
        
        self.f       = f
        self.f_prime = f_prime
        
    def deriv(self, x):
        
        return self.f_prime(x)
        
    def __call__(self, x):
        
        return self.f(x)
        
### --- Define sigmoid --- ###
        
def sigmoid_(z):
    
    return 1/(1+np.exp(-z))

def sigmoid_derivative(z):
    
    return sigmoid_(z)*(1-sigmoid_(z))

sigmoid = function(sigmoid_, sigmoid_derivative)
        

### --- Define sigmoid cross entropy --- ###

def sigmoid_cross_entropy_(z, y):
    
    p = sigmoid.f(z)
    
    return -np.mean(y*np.log(p) + (1 - y)*np.log(1 - p))

def sigmoid_cross_entropy_derivative(z, y):
    
    p = sigmoid.f(z)
    
    return (-y/p + (1 - y)/(1 - p))*sigmoid.f_prime(z)

sigmoid_cross_entropy = function(sigmoid_cross_entropy_, sigmoid_cross_entropy_derivative)

### --- Define ReLu --- ###

def relu_(h):
    
    return np.maximum(0, h)

def relu_derivative(h):
    
    return (h>0).astype(np.int32)

relu = function(relu_, relu_derivative)

### --- Define tanh --- ###

def tanh_(h):
    
    return np.tanh(h)

def tanh_derivative(h):
    
    return 1 - np.tanh(h)**2

tanh = function(tanh_, tanh_derivative)

### --- Define softmax --- ###

def softmax_(z):
    
    z = z - np.amax(z)
    
    return np.exp(-z)/np.sum(z)

def softmax_derivative(h):
    
    return 5
















    