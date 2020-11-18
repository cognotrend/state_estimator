# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 22:56:30 2020

@author: cognotrend
"""


def dd(i,j):
    if i==j:
        return 1
    else:
        return 0
    
def delta_meas(i,j,k):
    return dd(i,k)-dd(j,k)

def H_diff(num_devices,cols_per_device):
    H=np.zeros(1,num_devices*cols_per_device)
    cols = list(range(0,num_devices*cols_per_device))
    for j in cols:
        rem=j%cols_per_device
        if rem==0:
            H[0,j]=delta_meas()
    
    