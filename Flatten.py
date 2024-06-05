# -*- coding: utf-8 -*-
"""
Created on Thu May 30 17:22:14 2024

@author: Akash
"""
import numpy as np
from Layers.Base import BaseLayer


class Flatten(BaseLayer):
    def __init__(self): 
        super().__init__()
        self.input_tensor_shape = None
        
    def forward(self, input_tensor):
        self.input_tensor_shape = input_tensor.shape
        return np.reshape(input_tensor,(self.input_tensor_shape[0],-1))
    
    
    def backward(self, error_tensor):
        return np.reshape(error_tensor, (self.input_tensor_shape))