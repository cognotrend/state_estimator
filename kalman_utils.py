# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 22:56:30 2020

@author: cognotrend
"""
import numpy as np

def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)


def dd(i,j):
    if i==j:
        return 1
    else:
        return 0
    
def std_unit_colvec(index=0,size=3):
    suc = np.zeros((size,1))
    suc[index,0]=1
    return suc
    
def std_unit_rowvec(index=0, size=3):
    tmp = std_unit_colvec(index, size)
    sur = np.transpose(tmp)
    return sur


def delta_meas(i,j,k):
    return dd(i,k)-dd(j,k)

def H_diff(num_devices,ref_device,num_states_per_device=3):
    numcols = num_devices*num_states_per_device
    Hr=np.zeros((num_devices-1,num_devices))
    e_r = std_unit_rowvec(index=ref_device,size=num_devices)
    j=0
    for i in list(range(num_devices)):
        if i!=ref_device:
            Hr[j,:]=std_unit_rowvec(index=i, size=num_devices)-e_r
            j=j+1

    P = np.zeros((num_devices,num_devices*num_states_per_device))
    j=0
    for i in list(range(num_devices)):
#        print(i*num_states_per_device)
        P[i,:] = std_unit_rowvec(index=j,size=numcols)
        j=j+num_states_per_device
    H = Hr@P        
#    print(P)
    return H

    
    