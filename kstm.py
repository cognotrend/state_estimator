# kstm.py:  Kalman state transition matrix
import numpy as np
import math
class KalmanStateTransMatrix():
    def __init__(self,dt=1,tau_factor=3,state_size=3,phi_type=0):

# Kalman state transition matrix
        self.dt = dt
        if phi_type==0 and state_size==3:
            tau_m = tau_factor*self.dt  # for price measurement only
            beta = 1/tau_m
            beta_squared = math.pow(beta,2)
            rho_m = math.exp(-beta*self.dt)
            sigma_m = 1  # What is this for?  Not currently used!
            self.Phi = np.array([[1,self.dt,(1/beta_squared)*(-1 + beta*self.dt+rho_m)],\
                [0,1,(1/beta)*(1-rho_m)],\
                [0,0,rho_m]],\
               float)
        else:
            print('Using default STM')
            self.Phi = np.zeros((state_size,state_size))
            i=0
            while i<state_size:
                j=0
                while j<state_size:
                    if i==0:
                        self.Phi[(i,j)] = math.pow(self.dt,j)/math.factorial(j)
                    else:
                        if j<i:
                            self.Phi[(i,j)]=0
                        else:
                            self.Phi[(i,j)] = self.Phi[(i-1,j-1)]
                    j += 1
                i += 1


                    

