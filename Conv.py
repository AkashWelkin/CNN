# -*- coding: utf-8 -*-
"""
Created on Thu May 30 17:21:10 2024

@author: Akash
"""

from Layers.Base import BaseLayer
import numpy as np
import math

class Conv(BaseLayer):
    def __init__(self, stride_shape, convolution_shape, num_kernels): 
        super().__init__()
        self.trainable = True
        self.convolution_shape = convolution_shape
        self.num_kernels = num_kernels
        
        self.bias = np.random.uniform(0,1,self.num_kernels)
        
        self.weights = None # are not thing but weights
        
        self._gradient_weights = None
        self._gradient_bias = None
        
        # if only one value provided for both horizontal and vertical stride or else case
        if isinstance(stride_shape, int):
            self.stride_shape = (stride_shape, stride_shape)
        if len(stride_shape)>1:
            self.stride_shape = tuple(stride_shape)
        else:
            self.stride_shape = (stride_shape[0],stride_shape[0]) #===> if testcase fails then change it to int type as it is 1D stride
            
        # if 1D convolution layer then first part else second and third part for grey and color channel respectively.
        # compare it like number of kernels,number_of_channel,sequence_length -- 1D
        # compare it like number of kernels,number_of_channel,Height,Width    -- 2D
        
        if len(self.convolution_shape)==2:
            self.weights = np.random.uniform(0,1,(self.num_kernels,self.convolution_shape[0], self.convolution_shape[1]))
        elif len(self.convolution_shape)==3:
            self.weights = np.random.uniform(0,1,(self.num_kernels, self.convolution_shape[0],self.convolution_shape[1], self.convolution_shape[2]))
        
    @property
    def gradient_weights(self):
        return self._gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, gradient_weights):
        self._gradient_weights = gradient_weights
    
    @property
    def gradient_bias(self):
        return self._gradient_bias

    @gradient_bias.setter
    def gradient_bias(self, gradient_bias):
        self._gradient_bias = gradient_bias
    
    def zero_padding(self,input_tensor):
        
        if len(self.convolution_shape)==3:
            pad_h = (self.convolution_shape[1])//2
            pad_w = (self.convolution_shape[2])//2
            # batch,channel,height,width => padding values for all dimensions
            return np.pad(input_tensor,pad_width=((0,0),(0,0),(pad_h,pad_h),(pad_w,pad_w)),mode='constant',constant_values=0)
        else:
            pad_w = (self.convolution_shape[1])//2
            return np.pad(input_tensor,pad_width=((0,0),(0,0),(pad_w,pad_w)),mode='constant',constant_values=0)
    
    
    def calculate_output_shape(self,input_shape):
        
        # output dimension=((input_dimension + 2×padding−kernel size)/stride)+1        
        if len(self.convolution_shape)==3:
            b,c,y,x = input_shape
            output_height = math.ceil(y/self.stride_shape[0])
            output_width = math.ceil(x/self.stride_shape[1])
            
            #kernel_height = self.convolution_shape[1]
            #kernel_width = self.convolution_shape[2]
            #pad_h = (self.convolution_shape[1])//2
            #pad_w = (self.convolution_shape[2])//2
            #output_height = ((y + 2*(pad_h)-kernel_height)//self.stride_shape[0])+1
            #output_width = ((x + 2*(pad_w)-kernel_width)//self.stride_shape[1])+1
            #if output_width>x:
            #    output_width=x
            #if output_height>y:
            #    output_height=y
            return (b,self.num_kernels,output_height,output_width)
        else:
            b,c,y = input_shape
            output_width = math.ceil(y/self.stride_shape[0])
            #kernel_width = self.convolution_shape[1]
            
            #pad_w = (self.convolution_shape[1])//2
            #output_width = ((y + 2*(pad_w)-kernel_width)//self.stride_shape[0])+1
            return (b,self.num_kernels,output_width)
       
    def OneDConvComputation(self,input_tensor, input_tensor_padded, output_shape):
        batch_size,channels,y = input_tensor.shape
        kernel_width = self.convolution_shape[1]
        output_tensor = np.zeros(output_shape)
        for batch_itr in range(batch_size):
            for kernel in range(self.num_kernels):
                for idx in range(0,output_shape[2]):
                    start = idx * self.stride_shape[0]
                    end = start + kernel_width
                    for channel in range(channels):
                        output_tensor[batch_itr,kernel,idx] += np.sum(input_tensor_padded[batch_itr,channel,start:end]*self.weights[kernel,channel])
                    output_tensor[batch_itr,kernel,idx] += self.bias[kernel]
        return output_tensor
    
    def TwoDConvComputation(self,input_tensor,input_tensor_padded,output_shape):
        print("==>>>>>>>>>>>>>>>>>>>>{}".format(input_tensor.shape))
        batch_size,channels,y,x = input_tensor.shape
        kernel_height,kernel_width = self.convolution_shape[1],self.convolution_shape[2]
        output_tensor = np.zeros(output_shape)
        
        for batch_itr in range(batch_size):
            for kernel in range(self.num_kernels):
                for row in range(0,output_shape[2]):
                    for col in range(0,output_shape[3]):
                        y_start = row * self.stride_shape[0]
                        y_end = y_start + kernel_height
                        x_start = col * self.stride_shape[1] # this needs to verify and correct if testcase is Integer, means same for h and w
                        x_end = x_start + kernel_width
                        for channel in range(channels):
                            output_tensor[batch_itr,kernel,row,col] += np.sum(input_tensor_padded[batch_itr,channel,y_start:y_end,x_start:x_end] * self.weights[kernel,channel])
                        output_tensor[batch_itr,kernel,row,col] += self.bias[kernel]
        return output_tensor
        
    
    def forward(self, input_tensor):
        # for 1D b - batch,c - channels(conv_shape[0]),y - length(conv_shape[1])
        input_tensor_padded = self.zero_padding(input_tensor)
        print("newnewnew---------------->{}".format(input_tensor_padded.shape))
        output_shape = self.calculate_output_shape(input_tensor.shape)

        if len(self.convolution_shape)==2:
            return self.OneDConvComputation(input_tensor,input_tensor_padded,output_shape)
        
        elif len(self.convolution_shape)==3:
            return self.TwoDConvComputation(input_tensor, input_tensor_padded, output_shape)        
        
        
    
    def backward(self, error_tensor):
        pass
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    