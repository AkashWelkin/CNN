# -*- coding: utf-8 -*-
"""
Created on Sat May 18 14:45:31 2024

@author: Akash
"""

class BaseLayer:
    def __init__(self):
        self.trainable = False
        self.weights = None
        self.prediction_tensor=None
        self.loss=None
        self.error_tensor=None
        self.output_tensor=None
        self.input_tensor=None
        self.label_tensor=None
    

        