"""KalmanFilter class"""
#kalman_filter.py
import covariance as cov
import kstm
import process_noise as pn
import numpy as np
import math

default_numruns=5
default_q_factor=0.0001
default_meas_noise_sigma=0.1
class KalmanFilter():
    '''
    The KalmanFilter class implements a Kalman Filter, by default of order 3
    Proposed changes:  Implement a default measurement model based on a truth model that incorporates (by default)
    the process model and process model statistics, and executes a simulation using random noise arrays from numpy.
    Need separate classes for process model that provide a default process model.  Process noise should be embedded in the
    process model.  Process model should contain its own propagator / predictor. Process models should be compatible with
    ARIMA models produced by statsmodels package.
    Process models and measurement models should have their own separate timing models.
    KF state prediction / extrapolation update should be same as process model's time update.  In simulation mode,
    the KF should generate process noise inputs (or, better, the process model should generate its own).
    Both predictor and corrector methods cause the state and covariance to be updated.  The corrector causes the intial
    conditions for the next iteration to be reset.
    Add a ground truth simulator mode:  by revering ground truth out of measurments or
    Generating measurements from ground truth
    '''

    def __init__(self,meas_func=None,T=1,tau_factor=10,state_size=3,meas_size=1,
                 sigma=default_meas_noise_sigma,
                 num_blocks=1, # for ensemble of filters
                 phi_type=0,ref_model=1, num_runs=default_numruns):
        '''
        KalmanFilter object constructor
        '''
        if meas_func==None:
            self.meas_func = self.test_meas_func
        else:
            self.meas_func = meas_func
        self.numruns = num_runs
        self.T = T
        self.phi_type = phi_type
        self.tau_factor = tau_factor
        self.state_size = state_size
        self.Phi  = kstm.KalmanStateTransMatrix(T=self.T,tau_factor=self.tau_factor,state_size=self.state_size,phi_type=self.phi_type).Phi

        self.meas_size = meas_size
        self.sigma = sigma
        self.num_blocks = num_blocks
        self.ref_model = ref_model
        self.H = np.zeros((self.meas_size,self.state_size))
        if self.num_blocks>1:
            self.Phi = np.kron(self.Phi,np.eye(self.num_blocks))
            self.state_size = self.state_size*self.num_blocks
            self.meas_size = self.num_blocks
            self.H = np.zeros(1,self.state_size)
           # i=j=k=0
#            while i<self.state_size

        self.I = np.eye(self.state_size)
        # Create contstant Kalman process noise
        self.Q  = pn.ProcessNoise(T=self.T,tau_factor=self.tau_factor,
                                  state_size=self.state_size,
                                  phi_type=self.phi_type,
                                  q_factor=default_q_factor).Q
        # Constant measurement noise covariance matrix
        # Future:  Make time-varying as function of volatility
        self.R = cov.Covariance(size=self.meas_size,sigma1=self.sigma,msg='R measurement noise covariance matrix.').Cov
        # Measurement matrix:  Only price measurement
        self.H[0,0]= 1.0
        self.reset()
        print(self.__doc__)

    def reset(self):
        self.P_minus = cov.Covariance(size=self.state_size,sigma1=1.0,
                                    sigma2=0.1,sigma3=0.01,
                                    msg='Initial P_minus covariance matrix.').Cov
        self.P_minus_cum = np.zeros((self.state_size,self.state_size,self.numruns))
        self.P_minus_cum[:,:,0] = self.P_minus
        self.P_plus = cov.Covariance(self.state_size,1.0,0.1,0.01,msg='Initial P_plus covariance matrix.').Cov
        self.P_plus_cum = np.zeros((self.state_size,self.state_size,self.numruns))
        self.P_plus_cum[:,:,0] = self.P_plus
        self.x_minus = np.zeros((self.state_size,self.numruns))
        self.x_plus = np.zeros((self.state_size,self.numruns))
        meas_tmp = self.meas_func()
        if meas_tmp.size ==1:
            tmp_z = np.zeros((self.meas_size,1))
            tmp_z[0]=meas_tmp
            self.x_plus[0,0] = tmp_z
        else:
            self.x_plus[:,0] = meas_tmp

        self.K = np.zeros((self.state_size,self.meas_size,self.numruns))
        self.z = np.zeros((self.meas_size,self.numruns))
        self.zhat = np.zeros((self.meas_size,self.numruns))
        self.residual = self.z - self.zhat
        self.meas_array = np.random.randn(1,self.numruns)  # Scalar meas.
        self.k = 0

    def setNumRuns(self,numruns=default_numruns):
        old_numruns = self.numruns
        self.numruns=numruns
        print('Changing numruns from ',old_numruns,' to ',numruns)

    def run(self):
        self.reset()
        rsum=0
        rsumsq = 0
        while self.k<self.numruns-1:
            self.display()
            self.cycle()
            r = self.residual[0,self.k]
            rsum = rsum + r
            rsumsq = rsumsq + r*r
        self.display()
        avg = rsum/self.numruns
        mse = rsumsq/self.numruns
        rms = math.sqrt(mse)
        print('Avg residual: ',avg)
        print('Exponential of Avg residual: ',np.exp(avg))
        print('RMS of residual: ', rms)
        return self.x_plus, self.x_minus, self.residual

    def cycle(self):
        self.extrapolate()
        self.computeGain()
        self.update()

    def extrapolate(self):
        # Extapolate state and covariance:
        
        self.k = self.k+1
        self.x_minus[:,self.k] = np.dot(self.Phi,self.x_plus[:,self.k-1])
        self.P_minus  = np.dot(self.Phi,np.dot(self.P_plus,self.Phi.transpose())) + self.Q
        self.P_minus_cum[:,:,self.k] = self.P_minus

    def computeGain(self):
        S = np.dot(self.H,np.dot(self.P_minus,self.H.transpose()))+self.R
        S_inv = np.linalg.inv(S)
        self.K[:,:,self.k] = np.dot(np.dot(self.P_minus,self.H.transpose()),S_inv)

    def update(self):
        # Update Covariance
        self.P_plus = np.dot((self.I-np.dot(self.K[:,:,self.k],self.H)),\
                            np.dot(self.P_minus,\
                                   (self.I-np.dot(self.K[:,:,self.k],self.H)).transpose())) \
                     + np.dot(self.K[:,:,self.k],np.dot(self.R,self.K[:,:,self.k].transpose()))
        self.P_plus_cum[:,:,self.k] = self.P_plus
        meas_tmp = self.meas_func()
        if meas_tmp.size ==1:
            tmp_z = np.zeros((self.meas_size,1))
            tmp_z[0]=meas_tmp
            self.z[:,self.k] = tmp_z
        else:
            self.z[:,self.k] = meas_tmp
        self.zhat[:,self.k] = np.dot(self.H,self.x_minus[:,self.k])
        self.residual[:,self.k] = self.z[:,self.k] - self.zhat[:,self.k]
        weighted_residual = np.dot(self.K[:,:,self.k],self.residual[:,self.k])
        self.x_plus[:,self.k] = self.x_minus[:,self.k] + weighted_residual
#        new_state = self.x_plus[:,self.k]
#        print(new_state)
        

    def display(self):
        if self.k==0:
            print('\nStart\n')
            print('*************Initialization*******************')
            print('k: ',self.k)
            print('Prior State: (N/A)\n',self.x_minus[:,self.k])
            print('Prior Cov:\n', self.P_minus)
            print('Kalman Gain: (N/A)\n',self.K[:,:,self.k])
            print('Measurement: (N/A)\n',self.z[:,self.k])
            print('Pre-fit Residual:(N/A)\n',self.residual[:,self.k])
            print('Initial Posterior State:\n',self.x_plus[:,self.k])
            print('Initial Posterior Cov:\n', self.P_plus)
            print('**************End Initialization***************')
        else:            
            print('**********************************************')
            print('k: ',self.k)
            print('Prior State:\n',self.x_minus[:,self.k])
            print('Prior Cov:\n', self.P_minus)
            print('Kalman Gain:\n',self.K[:,:,self.k])
            print('Measurement:\n',self.z[:,self.k])
            print('Pre-fit Residual:\n',self.residual[:,self.k])
            print('Exponential of Residual:\n',np.exp(self.residual[:,self.k]))
            if self.x_minus[0,self.k] != 0.0:
                print('Residual as percent of State:\n',(self.residual[:,self.k]/self.x_minus[0,self.k])*100,'%')
            print('Posterior State:\n',self.x_plus[:,self.k])
            print('Posterior Cov:\n', self.P_plus)
            print('**********************************************')


    def test_meas_func(self):
        return self.meas_array[(0,self.k)]

