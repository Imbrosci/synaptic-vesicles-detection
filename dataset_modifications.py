#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 12:06:59 2020

Add Gaussian noise to tensors.

@author: barbara
"""
import torch

class AddGaussianNoise(object):
    def __init__(self,mean=0.,std=1.):
        self.mean=mean
        self.std=std
    
    def __call__(self,tensor):
        return tensor + torch.randn(tensor.size())*self.std+self.mean
    
    def __repr__(self):
        return self.__class__.name__ + '(mean={0}, std{1})'.format(self.mean,self.std)     
        
