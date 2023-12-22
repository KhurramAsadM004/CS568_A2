#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 01:32:15 2023

@author: Arbish Akram and Nazar Khan
"""

import numpy as np
from f_utils import *
import copy


def cal_numerical_gradient(self, l, m, n, train_X, train_t, eps):
    """
    Calculate the numerical gradient approximation for a specific weight 
    """
    self.parameters['W%s' % l][m][n] += eps # Perturb the weight slightly by adding epsilon    
    self.fprop(train_X)                     # Perform forward propagation 
    lplus = self.calculate_loss(train_t)    # Calculate the loss (lplus) with the perturbed weight
   
    self.parameters['W%s' % l][m][n] -= 2*eps   # Perturb the weight slightly by subtracting epsilon 
    self.fprop(train_X)                         # Perform forward propagation 
    lminus = self.calculate_loss(train_t)       # Calculate the loss (lminus) with the perturbed weight
    
    Numerical_grad = (lplus - lminus) / (2 * eps) # Calculate the numerical gradient approximation
    
    self.parameters['W%s' % l][m][n] += eps # Revert weight to its original value
   
    return Numerical_grad
          
    
def check_gradients(self, train_X, train_t): 
    """
    Check the gradients computed thourgh backpropagation against numerical gradient for each weight in the network
    """
    eps= 1e-7
    grad_ok = 0
    self.fprop(train_X) # Perform forward propagation 
    self.bprop(train_t) # Perform backward propagation to compute gradients
    
    mad = np.zeros(self.num_layers+1)
    for l in range(1, self.num_layers+1): 
        abs_diff = np.zeros(self.parameters['W%s' % l].shape);
        for m in range(0, self.num_neurons[l]):
            for n in range(0, self.num_neurons[l-1]):   
                
                # Retrieve the analytical gradient of weight n of neuron m in layer l
                Analytical_grad = self.grads['dW%s' % l][m][n] 
                # Analytical_grad /= self.mini_batch_size
                
                # Calculate the numerical gradient of weight n of neuron m in layer l
                Numerical_grad =  cal_numerical_gradient(self, l, m, n, train_X, train_t, eps)
                
                # Calculate the difference between the numerical and analytical gradients
                # print(Numerical_grad, Analytical_grad)# * self.mini_batch_size)
                diff = np.abs(Numerical_grad - Analytical_grad)
                abs_diff[m,n] = diff
                
                # Check if the difference exceeds the acceptable threshold (eps)
                if (diff> eps):
                    print("layer %s gradients are not ok, diff ="% l, diff)  
                    grad_ok = 0
                    return grad_ok
                else:
                    print("layer %s gradients are ok"% l)
        
        mad[l]=np.max(abs_diff).item()

    if np.max(mad)>eps:
        grad_ok = 0
    else:
        grad_ok = 1

    return grad_ok
