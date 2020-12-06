#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 22:44:09 2020

@author: mcarroll
"""
import numpy as np

def array_test():
    y=np.zeros((2,1))
    print('y shape: '+str(y.shape))
    return y

x = array_test()
print('x shape: '+str(x.shape))
