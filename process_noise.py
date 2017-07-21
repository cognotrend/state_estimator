import numpy as np
import math
class ProcessNoise():
    def __init__(self,T=1,tau_factor=10,state_size=3,Q0=[1],phi_type=0):
        if phi_type==0 and state_size==3:
            self.T = T
            Tau_m = tau_factor*self.T  # for price measurement only
            T=self.T
            beta = 1/Tau_m
            self.rho_m = math.exp(-beta*T)
            self.sigma_m = 1
#print(rho_m)
# Kalman state transition matrix
            beta_squared = math.pow(beta,2)
            beta_squared = math.pow(beta,2)
            beta_cubed = math.pow(beta,3)
            beta_fourth = math.pow(beta,4)
            beta_fifth = math.pow(beta,5)
            T_squared = math.pow(T,2)
            T_cubed = math.pow(T,3)
            self.q11 = (1/(2*beta_fifth))*(1-math.exp(-2*beta*T)+2*beta*T+2*beta_cubed*T_cubed/3-2*beta_squared*T_squared-4*beta*T*math.exp(-beta*T))
            self.q12 = (1/(2*beta_fourth))*(math.exp(-2*beta*T)+1-2*self.rho_m+2*beta*T*self.rho_m-2*beta*T+beta_squared*T_squared)
            self.q13 = (1/(2*beta_cubed))*(1-math.exp(-2*beta*T)-2*beta*T*self.rho_m)
            self.q22 = (1/(2*beta_cubed))*(4*self.rho_m-3-math.exp(-2*beta*T)+2*beta*T)
            self.q23 = (1/(2*beta_squared))*(math.exp(-2*beta*T)+1-2*self.rho_m)
            self.q33 = (1/(2*beta))*(1 - math.exp(-2*beta*T))
            self.Q1 = np.array([[self.q11, self.q12, self.q13],[self.q12,self.q22,self.q23],[self.q13, self.q23, self.q33]],float)
            self.Q  = (2*math.pow(self.sigma_m,2)/Tau_m)*self.Q1
        elif Q0==[1]:
            self.Q = np.eye(state_size)
        else:
            self.Q = Q0

