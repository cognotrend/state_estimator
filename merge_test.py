# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 12:46:33 2020

@author: cognotrend
"""


import state_vector as sv
import numpy as np

sv1=sv.StateVector(state_names=np.array(['State a','b','c']))
sv2=sv.StateVector(state_names=np.array(['State dddd','e','f']))
sv=sv.StateVector()

sv.merge(sv1,sv2)


