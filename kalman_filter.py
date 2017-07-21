#kalman_filter.py
import covariance as cov
import kstm
import process_noise as pn
import numpy as np
import math
class KalmanFilter():
    def __init__(self,meas_func=None,T=1,tau_factor=10,state_size=3,meas_size=1,sigma=0.1,num_blocks=1,
                 phi_type=0,ref_model=1):
        """

        :type ref_model: int
        :type num_blocks: int
        """
        if meas_func==None:
            self.meas_func = self.test_meas_func
        else:
            self.meas_func = meas_func
        self.T = T
        self.phi_type = phi_type
        self.tau_factor = tau_factor
        self.state_size = state_size
        self.meas_size = meas_size
        self.sigma = sigma
        self.num_blocks = num_blocks
        self.ref_model = ref_model
        self.Phi  = kstm.KalmanStateTransMatrix(self.T,self.tau_factor,self.state_size,phi_type=self.phi_type).Phi
        self.H = np.zeros((self.meas_size,self.state_size))
        if self.num_blocks>1:
            self.Phi = np.kron(self.Phi,np.eye(self.num_blocks))
            self.state_size = self.state_size*self.num_blocks
            self.meas_size = self.num_blocks
            self.H = np.zeros(1,self.state_size)
            i=j=k=0
#            while i<self.state_size

        self.I = np.eye(self.state_size)
        # Create contstant Kalman process noise
        self.Q  = pn.ProcessNoise(T,tau_factor,self.state_size,self.phi_type).Q
        # Constant measurement noise covariance matrix
        self.R = cov.Covariance(size=self.meas_size,sigma1=self.sigma).Cov
        # Measurement matrix:  Only price measurement
        self.H[0,0]= 1.0
        self.reset()

    def cycle(self):
        self.computeGain()
        self.update()
        self.extrapolate()

    def display(self):
        print('k: ',self.k)
        print('Old State:\n',self.x_old)
        print('Old Cov:\n', self.P_old)
        print('Kalman Gain:\n',self.K)
        print('Measurement:\n',self.z)
        print('Residual:\n',self.residual)
        print('New State:\n',self.x_new)
        print('New Cov:\n', self.P_new)
    def run(self,numruns=10):
        i=0
        rsum=0
        rsumsq = 0
        while i<numruns:
            self.display()
            self.cycle()
            i=i+1
            r = self.residual[0]
            rsum = rsum + r
            rsumsq = rsumsq + r*r
        avg = rsum/numruns
        mse = rsumsq/numruns
        rms = math.sqrt(rsumsq)
        print('Avg residual: ',avg)
        print('RMS of residual: ', rms)

    def computeGain(self):
        S = np.dot(self.H,np.dot(self.P_old,self.H.transpose()))+self.R
        S_inv = np.linalg.inv(S)
        self.K = np.dot(np.dot(self.P_old,self.H.transpose()),S_inv)

    def update(self):
        # Update Covariance
        self.P_new = np.dot((self.I-np.dot(self.K,self.H)),\
                            np.dot(self.P_old,\
                                   (self.I-np.dot(self.K,self.H)).transpose())) \
                     + np.dot(self.K,np.dot(self.R,self.K.transpose()))
        meas_tmp = self.meas_func()
        if meas_tmp.size ==1:
            tmp_z = np.zeros((self.meas_size,1))
            tmp_z[0]=meas_tmp
            self.z = tmp_z
        else:
            self.z = meas_tmp
        if self.k==0:
            self.zhat = self.z
        else:
            self.zhat = np.dot(self.H,self.x_old)
        self.residual = self.z - self.zhat
        self.x_new = self.x_old + np.dot(self.K,self.residual)
        
    def extrapolate(self):
        # Extapolate state and covariance:
        self.x_old = np.dot(self.Phi,self.x_new)
        self.P_old  = np.dot(self.Phi,np.dot(self.P_new,self.Phi.transpose())) + self.Q
        self.k=self.k+1

    def reset(self):
        self.P_old = cov.Covariance(size=self.state_size,sigma1=1.0,sigma2=0.1,sigma3=0.01).Cov
        self.P_new = cov.Covariance(self.state_size,1.0,0.1,0.01).Cov
        self.x_old = np.zeros((self.state_size,1))
        self.x_old[1] = 0
        self.x_new = np.zeros((self.state_size,1))
        self.K = np.zeros((self.state_size,self.meas_size))
        self.z = np.zeros((self.meas_size,1))
        self.zhat = np.zeros((self.meas_size,1))
        self.residual = self.z - self.zhat
        self.meas_array = np.random.randn(1,10)
        self.k = 0

    def test_meas_func(self):
        return self.meas_array[(0,self.k)]

