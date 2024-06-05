# -*- coding: utf-8 -*-
"""
Created on Thu May 30 17:22:53 2024

@author: Akash
"""
import numpy as np
from Layers.Base import BaseLayer

class Sgd():
    def __init__(self, learning_rate):
        self.learning_rate = float(learning_rate)
        
    def calculate_update(self,weight_tensor,gradient_tensor):
        self.weightTensor = np.array(weight_tensor)
        self.gradientTensor = np.array(gradient_tensor)
        self.updatedWeight = self.weightTensor - self.learning_rate*(self.gradientTensor)
        return np.array(self.updatedWeight)

class SgdWithMomentum:
    def __init__(self,learning_rate, momentum_rate):
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.momentum_v = None
        
    def calculate_update(self, weight_tensor, gradient_tensor):
        if self.momentum_v is None:
            self.momentum_v = np.zeros_like(weight_tensor)
        self.momentum_v = self.momentum_rate*self.momentum_v - self.learning_rate*gradient_tensor
        return weight_tensor + self.momentum_v
            
        
    
class Adam:
    def __init__(self,learning_rate, mu, rho):
        # mu = B1 and rho = B2
        self.learning_rate = learning_rate
        self.mu = mu
        self.rho = rho
        self.momentum_v = None # from slides, this is first momentum estimation
        self.momentum_p = None # This is second momentum estimation
        self.exponent_k = 1
        self.eps = 1e-8 # don't change this as this is the comment recommendation in testcases, np.finfo().eps won't work here.
        
    def calculate_update(self, weight_tensor, gradient_tensor):
        if self.momentum_v is None: 
            self.momentum_v = np.zeros_like(weight_tensor)
        if self.momentum_p is None:
            self.momentum_p = np.zeros_like(weight_tensor)
        
        self.momentum_v = self.mu*self.momentum_v + (1-self.mu)*gradient_tensor
        self.momentum_p = self.rho*self.momentum_p + (1-self.rho)*(np.square(gradient_tensor))
        
        # bias correction
        corrected_momentum_v = self.momentum_v/(1-self.mu**self.exponent_k)
        
        corrected_momentum_p = self.momentum_p/(1-self.rho**self.exponent_k)
        
        updated_weight_tensor = weight_tensor - (self.learning_rate * (corrected_momentum_v/(np.sqrt(corrected_momentum_p)+self.eps)))
        
        self.exponent_k+=1 # this is required as it will change with next layer
        return updated_weight_tensor
            
            
            
            
            
            
            
            
            
            
            
            
            