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
        collated = np.empty((2*m,),dtype='object')
#        collated = collated.astype(type(v1[0]))
        j=0
        for i in range(m):
            collated[j]=v1[i]
            collated[j+1]=v2[i]
            j=j+2
    else:
        collated=-1
    return collated

def new_collate(v1,v2):
    if v1.shape==v2.shape:
        (n,m)=v1.shape
        tmp = np.empty((2,n),dtype='object')
        tmp[0,:]=np.transpose(v1[:,0])
        tmp[1,:]=np.transpose(v2[:,0])
        tmp=np.transpose(tmp)
        collated=tmp.reshape((2*n,1))
        return collated
    else:
        return -1

def std_sawtooth_plot(kfobj,fignum=1,expflag=0,last_percent=0.10,
                      title_prefix=''): #first three state variables
    plt.style.use('seaborn')
    plt.figure(fignum)
    lastnum=int(kfobj.numruns*last_percent)
    start = kfobj.numruns - lastnum
    v1 = np.zeros((3,kfobj.numruns-start))
    v2 = np.zeros((3,kfobj.numruns-start))
    sawtooth = np.zeros((3,2*(kfobj.numruns-start)))
    for i in [0,1,2]:
        v1[i,:] = kfobj.P_minus_cum[i,i,-lastnum:kfobj.numruns].reshape(lastnum,)
        v2[i,:] = kfobj.P_plus_cum[i,i,-lastnum:kfobj.numruns].reshape(lastnum,)
        doubletime=collate(np.arange(0,lastnum),np.arange(0,lastnum))
        sawtooth[i,:]=collate(v1[i,:],v2[i,:])
    if expflag==1:
        sawtooth=np.exp(2*sawtooth)
    ax0=plt.subplot(3,1,1)
    plt.plot(doubletime,sawtooth[0,:])
    plt.setp(ax0.get_xticklabels(), visible=False)
    plt.title(title_prefix+'Covariance')
    
    ax1=plt.subplot(3,1,2,sharex=ax0) 
    plt.plot(doubletime,sawtooth[1,:])
    plt.setp(ax1.get_xticklabels(), visible=False)

    ax2=plt.subplot(3,1,3,sharex=ax1) 
    plt.plot(doubletime,sawtooth[2,:])
    plt.xlabel('Epochs')
    plt.tight_layout()
    plt.show()

def plot_residuals(kfobj,fignum=2, 
            expflag=0, 
            title_prefix='',
            legend_str = ['']):
    data1 = kfobj.zhat[:,0,:]
    data2 = kfobj.z[:,0,:]
    data3 = kfobj.residual[:,0,:]
    suffix=''
    if expflag==1:
        data1 = np.exp(data1)
        data2 = np.exp(data2)
        data3 = kfobj.exp_residual[:,0,:]
        suffix = ' (Exp)'

    plt.style.use('seaborn')
    epochs=list(range(0,kfobj.numruns))
    if kfobj.meas_size==1:
        fig, axs = plt.subplots(nrows=kfobj.meas_size,ncols=1, sharex=True)
        myfig=[]
        myfig.append(fig)
        myaxs=[]
        myaxs.append(axs)
    else:
        myfig, myaxs = plt.subplots(nrows=kfobj.meas_size,ncols=1, sharex=True)
        
    for i in list(range(0,kfobj.meas_size)):
        if i==0:
            myaxs[i].set_title('Filter Residuals for '+kfobj.filter_id)
        myaxs[i].plot(epochs[1:kfobj.numruns],data3[i,1:kfobj.numruns],
                    label='Residuals: '+str(i))
        myaxs[i].legend(loc='upper left')
    plt.tight_layout() 
    plt.show()

#    ax1=plt.subplot(3,1,2,sharex=ax2)
#    plt.setp(ax1.get_xticklabels(), visible=False)
#    plt.plot(epochs[1:kfobj.numruns],data2.reshape((kfobj.numruns,))[1:kfobj.numruns])
#    plt.title(title_prefix+'Actual Measurements'+suffix,fontsize=10)
#    plt.show()
    
    # plt.figure(fignum+1)
    # plt.subplot(3,1,1)
    # plt.hist(np.transpose(kfobj.residual),bins=100)
    # plt.title(title_prefix+'Residuals Histogram')
    # plt.subplot(3,1,2)
    # plt.hist(np.exp(np.transpose(kfobj.residual)),bins=100)
    # plt.title('Exponential of Residuals')
    # plt.subplot(3,1,3)
    # mu=np.mean(np.transpose(kfobj.exp_residual))
    # mu=round(mu,2)
    # sigma=np.std(np.transpose(kfobj.exp_residual))
    # sigma=round(sigma,2)
    # plt.hist(np.transpose(kfobj.exp_residual),bins=100,label=r'$\mu$=' + str(mu)+ ', $\sigma$='+str(sigma))
    # plt.legend()
    # plt.title('Residuals of Exponentials')
    # plt.show()
  
def plot_gains(kfobj,state=0,fignum=4):
    plt.style.use('seaborn')
    epochs=np.arange(kfobj.numruns)
    fig, axs = plt.subplots(nrows=kfobj.state_size,ncols=1, sharex=True)
    for i in list(range(0,kfobj.state_size)):
        gains = kfobj.K_cum[i,0,:].transpose()
        axs[i].plot(epochs[1:kfobj.numruns],gains[1:kfobj.numruns],
                    label='Gain: State '+str(i)+',Meas '+str(0))
        axs[i].legend(loc='upper left')
        if i==0:
            axs[i].set_title('Kalman Gains for each state')

    plt.tight_layout()
    plt.show()

def plot_states(kfobj,state=0,fignum=4):
    epochs=np.arange(kfobj.numruns)
    fig, axs = plt.subplots(nrows=kfobj.state_size,ncols=1, sharex=True)
    for i in list(range(0,kfobj.state_size)):
        states = np.exp(kfobj.x_plus[i,:].transpose())
        axs[i].plot(epochs[1:kfobj.numruns],states[1:kfobj.numruns],
                    label='State '+str(i))
        axs[i].legend(loc='upper left')

        if i==0:
            axs[i].set_title('State Estimates for '+kfobj.filter_id)

    plt.tight_layout()
    plt.show()


def plot_posgains(kfobj,fignum=3, expflag=0):
    plt.figure(fignum)
    data = kfobj.posgains
    if expflag==1:
        data = np.exp(data)
    plt.plot(data)
    plt.title('Positive Gains')

    