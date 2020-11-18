# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 12:10:58 2020
State Vector Class
@author: cognotrend
"""
import numpy as np
import kalplots as kp

class StateVector():
    def __init__(self,
                 state_size=3,
                 state_names=np.empty((3,1),dtype=object)
                 ):
        self.state_size = state_size
        self.state_vector = np.zeros((self.state_size,1))
        self.state_names = state_names.reshape(self.state_size,1)
        
    def merge(self,sv1,sv2):
        if sv1.state_size==sv2.state_size:
            self.state_size = 2*sv1.state_size
            self.state_vector = np.zeros((sv1.state_size+sv2.state_size,1))
            self.state_names = kp.new_collate(sv1.state_names,sv2.state_names)
        
        
        
    
                 

