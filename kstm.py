# kstm.py:  Kalman state transition matrix
import numpy as np
import math
class KalmanStateTransMatrix():
    def __init__(self,T=1,tau_factor=10,state_size=3):
        tau_m = tau_factor*T  # for price measurement only
        beta = 1/tau_m
        beta_squared = math.pow(beta,2)
        rho_m = math.exp(-beta*T)
        sigma_m = 1  # What is this for?  Not currently used!

# Kalman state transition matrix
        if state_size==3:
            self.Phi = np.array([[1,T,(1/beta_squared)*(-1 + beta*T+rho_m)],\
                [0,1,(1/beta)*(1-rho_m)],\
                [0,0,rho_m]],\
               float)
        else:
            self.Phi = np.eye(state_size)

