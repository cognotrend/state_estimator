# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 18:34:27 2020

@author: cognotrend
"""
import numpy as np
import matplotlib.pyplot as plt

def collate(v1,v2):
# collate.m collates two vectors of length m into a single vector of
# length 2m.
    if v1.shape==v2.shape:
        (m,)=v1.shape
        collated = np.zeros((2*m,))
        j=0
        for i in range(m):
            collated[j]=v1[i]
            collated[j+1]=v2[i]
            j=j+2
    else:
        collated=-1
    return collated


def std_sawtooth_plot(kfobj,fignum=1,): #first three state variables
    plt.figure(fignum)
    v1 = np.zeros((3,kfobj.numruns))
    v2 = np.zeros((3,kfobj.numruns))
    sawtooth = np.zeros((3,2*kfobj.numruns))
    for i in [0,1,2]:
        v1[i,:] = kfobj.P_minus_cum[i,i,:].reshape(kfobj.numruns,)
        v2[i,:] = kfobj.P_plus_cum[i,i,:].reshape(kfobj.numruns,)
        doubletime=collate(np.arange(0,kfobj.numruns),np.arange(0,kfobj.numruns))
        sawtooth[i,:]=collate(v1[i,:],v2[i,:])

    ax0=plt.subplot(3,1,1)
    plt.plot(doubletime,sawtooth[0,:])
    plt.setp(ax0.get_xticklabels(), visible=False)
    plt.title('Covariance')
    
    ax1=plt.subplot(3,1,2,sharex=ax0) 
    plt.plot(doubletime,sawtooth[1,:])
    plt.setp(ax1.get_xticklabels(), visible=False)

    ax2=plt.subplot(3,1,3,sharex=ax1) 
    plt.plot(doubletime,sawtooth[2,:])
    plt.xlabel('Epochs')
    plt.show()
