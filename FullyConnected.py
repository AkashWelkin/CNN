# -*- coding: utf-8 -*-
"""
Created on Sat May 18 15:12:33 2024

@author: Akash
"""
import numpy as np
'''
import os
import sys 


module_path = os.path.abspath(os.path.join('..',r'E:\MS_AI\Sem-II\DL\exercise\exercise1_material\src_to_implement'))
sys.path.append(module_path)
'''
from Layers.Base import BaseLayer
from Layers.Initializers import *

class FullyConnected(BaseLayer):
    def __init__(self,input_size,output_size):
        super().__init__()
        self.trainable = True
        self.batch_size = None
        self._gradient_weights = None
        self.input_size = input_size
        self.output_size = output_size
        self.bias = None # Initialization in initialize method
        self.weights = None # Refactored the weight initialization funcation call, initialize()
        self._optimizer = None
        
    def forward(self,input_tensor):
        #self.bias = np.ones((input_tensor.shape[0],1)) commenting/remove later as already initialising bias & weight
        self.input_tensor = np.concatenate((input_tensor,self.bias),axis=1)
        self.output_tensor = np.dot(self.input_tensor,self.weights)
        return self.output_tensor
    
    @property
    def optimizer(self):
        return self._optimizer
    
    @optimizer.setter
    def optimizer(self,optimizer):
        self._optimizer = optimizer
        
    
    def backward(self,error_tensor):
        self._gradient_weights = np.dot(self.input_tensor.T, error_tensor)
           
        # Propagate error tensor to previous layer
        error_tensor_prev = np.dot(error_tensor, self.weights.T)
        error_tensor_prev = error_tensor_prev[:, :-1]  # Remove the bias part

        # Update weights using the optimizer if it is set
        if self._optimizer:
            self.weights = self._optimizer.calculate_update(self.weights, self._gradient_weights)
       
        return error_tensor_prev
        
    @property
    def gradient_weights(self):
        return self._gradient_weights
    
    def initialize(self,weights_initializer, bias_initializer):
        self.weights = weights_initializer.initialize((self.input_size,self.output_size),self.input_size,self.output_size)
        self.bias = bias_initializer.initialize((self.output_size,),self.input_size,self.output_size)
        
        
        
        
        