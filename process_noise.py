import numpy as np
import math
'''
Need to modify to take into account empirical  PSDs
See:  "Process and Measurement Noise Estimation for Kalman Filtering"
Yalcin Bulut, D. Vines-Cavanaugh, Dionisio Bernal, 2010 

'''
class ProcessNoise():
    def __init__(self,dt=1,tau_factor=1,state_size=3,Q0=[1],phi_type=0,q_factor=1):
        self.dt = dt
        self.alt_dt = 3*self.dt
        Tau_m = tau_factor*self.dt  # for price measurement only
        if phi_type==0 and state_size==3:
            beta = 1/Tau_m
            self.rho_m = math.exp(-beta*self.dt)
            self.sigma_m = 0.001
#print(rho_m)
# Kalman state transition matrix
            beta_squared = math.pow(beta,2)
            beta_squared = math.pow(beta,2)
            beta_cubed = math.pow(beta,3)
            beta_fourth = math.pow(beta,4)
            beta_fifth = math.pow(beta,5)
            T_squared = math.pow(self.dt,2)
            T_cubed = math.pow(self.dt,3)
            self.q11 = (1/(2*beta_fifth))*(1-math.exp(-2*beta*self.dt)
                                           +2*beta*self.dt
                                           +2*beta_cubed*T_cubed/3
                                           -2*beta_squared*T_squared
                                           -4*beta*self.dt*math.exp(-beta*self.dt))
            self.q12 = (1/(2*beta_fourth))*(math.exp(-2*beta*self.dt)+1
                                            -2*self.rho_m
                                            +2*beta*self.dt*self.rho_m
                                            -2*beta*self.dt+beta_squared*T_squared)
            self.q13 = (1/(2*beta_cubed))*(1-math.exp(-2*beta*self.dt)-2*beta*self.dt*self.rho_m)
            self.q22 = (1/(2*beta_cubed))*(4*self.rho_m-3-math.exp(-2*beta*self.dt)+2*beta*self.dt)
            self.q23 = (1/(2*beta_squared))*(math.exp(-2*beta*self.dt)+1-2*self.rho_m)
            self.q33 = (1/(2*beta))*(1 - math.exp(-2*beta*self.dt))
            self.Q1 = np.array([[self.q11, self.q12, self.q13],[self.q12,self.q22,self.q23],[self.q13, self.q23, self.q33]],float)
            self.Q  = (2*math.pow(self.sigma_m,2)/Tau_m)*self.Q1
            self.Alt_Q = self.Q  # temporary
        elif Q0==[1] and state_size==3:  #Standard "Newton/Taylor" process noise
            self.Q = np.eye(state_size)  #See Grewal & Andrews pg. 147
            self.Q[0,0]=math.pow(self.dt,5)/20
            self.Q[0,1]=math.pow(self.dt,4)/8
            self.Q[0,2]=math.pow(self.dt,3)/6

            self.Q[1,0]=math.pow(self.dt,4)/8
            self.Q[1,1]=math.pow(self.dt,3)/3
            self.Q[1,2]=math.pow(self.dt,2)/2

            self.Q[2,0]=math.pow(self.dt,3)/6
            self.Q[2,1]=math.pow(self.dt,2)/2
            self.Q[2,2]=self.dt
            self.Q = q_factor*self.Q

            self.Alt_Q = np.eye(state_size)  #See Grewal & Andrews pg. 147
            self.Alt_Q[0,0]=math.pow(self.alt_dt,5)/20
            self.Alt_Q[0,1]=math.pow(self.alt_dt,4)/8
            self.Alt_Q[0,2]=math.pow(self.alt_dt,3)/6

            self.Alt_Q[1,0]=math.pow(self.alt_dt,4)/8
            self.Alt_Q[1,1]=math.pow(self.alt_dt,3)/3
            self.Alt_Q[1,2]=math.pow(self.alt_dt,2)/2

            self.Alt_Q[2,0]=math.pow(self.alt_dt,3)/6
            self.Alt_Q[2,1]=math.pow(self.alt_dt,2)/2
            self.Alt_Q[2,2]=self.alt_dt
            self.Alt_Q = q_factor*self.Alt_Q
        else:
            self.Q = Q0

