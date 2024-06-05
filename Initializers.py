# -*- coding: utf-8 -*-
"""
Created on Thu May 30 15:25:26 2024

@author: Akash
"""
import numpy as np

class Constant:
    def __init__(self,ConstWeightInit=0.1):
        self.ConstWeightInit = ConstWeightInit
        
    
    def initialize(self, weights_shape, fan_in, fan_out):
        return np.ones(weights_shape)*self.ConstWeightInit

class UniformRandom:
    def __init__(self):
        pass
    
    def initialize(self, weights_shape, fan_in, fan_out):
        return np.random.uniform(0,1,weights_shape)

class Xavier:
    def __init__(self):
        pass
    
    def initialize(self, weights_shape, fan_in, fan_out):
        std = np.sqrt(2/(fan_in+fan_out))
        return np.random.normal(0,std,weights_shape)

class He:
    def __init__(self):
        pass
    
    def initialize(self, weights_shape, fan_in, fan_out):
        std = np.sqrt(2/fan_in)
        return np.random.normal(0,std,weights_shape)
    