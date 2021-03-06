"""KalmanFilter class"""
#kalman_filter.py
import covariance as cov
import kstm
import process_noise as pn
import numpy as np
import math
import datetime
import kalman_utils as ku

default_numruns=-1
default_q_factor=.01
default_meas_noise_sigma=0.1
default_logmode = 0
deterministic = False
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
    Add module to read alphavantage csv files with pandas.  Done.
    To do 2020-11-09:  Integrate block measurements with block state (ensemble) Done
    x is nx1, w is nx1, Phi is nxn, z is num_blocks-1x1, v is num_blocks-1xn,
    H is num_blocksxn,  P is nxn, Q is nxn, R is num_blocks-1 x num_blocksxn
    2) Make this event-driven based on obtaining a measurment, with extrapolaton from 
    previous update only done when a new measurement is obtained.  Use timedelta for dt.  
    3) Possibly extrapolate via small amounts in between measurements.  
    4) Make initial update estimate same as measurement (zero residual)
    5) Use Corey Schafer's subplot approach

    '''

    def __init__(self,
                 dt=1,
                 tau_factor=1,
                 basic_state_size=3,
                 meas_size=1,
                 init_state_val=0.0,
                 meas_obj=None,
                 meas_func=None,
                 sigma=default_meas_noise_sigma,
                 sigma1=1.0,
                 sigma2=0.1,
                 sigma3=0.01,
                 num_blocks=1, # for ensemble of filters
                 composite=False,
                 phi_type=0,ref_model=1, 
                 logmode=default_logmode,
                 num_runs=default_numruns,
                 displayflag=True,
                 verbose=False,
                 q_factor = default_q_factor,
                 filter_id='Test',
                 epoch_dumps=-1
                 ):
        '''
        KalmanFilter object constructor
        '''
        self.sigma1=sigma1
        self.sigma2=sigma2
        self.sigma3=sigma3
        self.filter_id = filter_id
        self.epoch_dumps = epoch_dumps
        self.composite = composite
        self.logmode = logmode
        self.displayflag = displayflag
        self.verbose = verbose
# Set up Process Model (should be object):
        self.dt = dt
        self.phi_type = phi_type
        self.tau_factor = tau_factor
        self.init_state_val = init_state_val

        self.num_blocks = num_blocks
        self.basic_state_size = basic_state_size
        if self.num_blocks>1:
            self.state_size = self.num_blocks*self.basic_state_size
        else:
            self.state_size = self.basic_state_size
        
        # Create State Transition Matrix  (should be subtask under ProcessModel)
        self.kstmobj = kstm.KalmanStateTransMatrix(dt=self.dt,
                                                tau_factor=self.tau_factor,
                                                basic_state_size=self.basic_state_size,
                                                phi_type=self.phi_type)
        self.Basic_Phi  = self.kstmobj.Phi  #basic size
        self.Basic_Alt_Phi = self.kstmobj.Alt_Phi  # Alt_Phi uses longer dt (currently 1.5)

        self.I = np.eye(self.state_size)

# Create contstant process noise covariance Q:
        self.pnobj  = pn.ProcessNoise(dt=self.dt,
                                tau_factor=self.tau_factor,
                                state_size=self.basic_state_size,
                                phi_type=self.phi_type,
                                q_factor=default_q_factor)
        self.Basic_Q = self.pnobj.Q
        self.Basic_Alt_Q = self.pnobj.Alt_Q
        if deterministic:
            self.Basic_Q = np.zeros((self.basic_state_size,self.basic_state_size))
            self.Basic_Alt_Q = np.zeros((self.basic_state_size,self.basic_state_size))
            

# For ensembling identical KFs into a single KF:
        if self.num_blocks>1:
            self.Phi = np.kron(np.eye(self.num_blocks),self.Basic_Phi)
            self.Alt_Phi = np.kron(np.eye(self.num_blocks),self.Basic_Alt_Phi)
            self.Q = np.kron(np.eye(self.num_blocks),self.Basic_Q)
            self.Alt_Q = np.kron(np.eye(self.num_blocks),self.Basic_Alt_Q)
            self.num_states_per_block = self.basic_state_size
            self.meas_size = self.num_blocks-1
        else:
            self.Phi = self.Basic_Phi
            self.Alt_Phi = self.Basic_Alt_Phi
            self.Q   = self.Basic_Q
            self.Alt_Q = self.Basic_Alt_Q


# Set up Measurement Model
# Constant measurement noise covariance matrix
# Set up MeasurementModel (should be distinct class of objects):
        self.meas_obj = meas_obj
        if meas_func==None:
            self.meas_func = self.test_meas_func
        else:
            self.meas_func = meas_func
        self.meas_size = meas_size
        self.sigma = sigma
        self.num_blocks = num_blocks
        self.ref_model = ref_model
        if num_runs == -1:
            if meas_obj != None:
                self.numruns = meas_obj.num_meas-1
            else:
                self.numruns = default_numruns
        else:
            self.numruns = num_runs
          
        
        self.z = np.zeros((self.meas_size,1,self.numruns))
        self.zhat = np.zeros((self.meas_size,1,self.numruns))
        # Future:  Make time-varying as function of volatility
        self.R = cov.Covariance(size=self.meas_size,
                                sigma1=self.sigma,
                                msg='R measurement noise covariance matrix.').Cov
        # Measurement matrix:  Only state  0 (price) measurement
        if self.num_blocks>1:
            ref_device=0  # May have to loop through these when doing update
            self.H=ku.H_diff(self.num_blocks,ref_device,self.num_states_per_block)
        else:
            self.H = np.zeros((self.meas_size,self.state_size))
            self.H[0,0]= 1.0
            if self.composite:
                numblocks = int(self.meas_size/self.basic_state_size)
                for i in list(range(0,numblocks)):
                    self.H[i*self.basic_state_size:(i+1)*self.basic_state_size,:]=np.eye(self.basic_state_size)
                    
            

        self.Basic_P = cov.Covariance(size=self.basic_state_size,
                                       sigma1=self.sigma1,
                                       sigma2=self.sigma2,
                                       sigma3=self.sigma3,
                                    msg='Initial P_minus covariance matrix.').Cov
        if self.num_blocks>1:
            self.P_minus = np.kron(np.eye(self.num_blocks),self.Basic_P)
        else:
            self.P_minus = self.Basic_P

        self.K = np.zeros((self.state_size,self.meas_size))

        self.reset()
#        print(self.__doc__)

    def reset(self):
        self.Basic_P = cov.Covariance(size=self.basic_state_size,
                                       sigma1=self.sigma1,
                                       sigma2=self.sigma2,
                                       sigma3=self.sigma3,
                                    msg='Reset P_minus covariance matrix.').Cov
        if self.num_blocks>1:
            self.P_minus = np.kron(np.eye(self.num_blocks),self.Basic_P)
        else:
            self.P_minus = self.Basic_P

        self.P_minus_cum = np.zeros((self.state_size,self.state_size,self.numruns))
        self.P_minus_cum[:,:,0] = self.P_minus

        self.P_plus = self.P_minus

        self.P_plus_cum = np.zeros((self.state_size,self.state_size,self.numruns))
        self.P_plus_cum[:,:,0] = self.P_plus
        
        self.x_minus = np.zeros((self.state_size,self.numruns))
#        self.x_minus[0,0] = self.x_minus[0,0]+np.random.normal(0,self.sigma)

        self.x_plus = np.zeros((self.state_size,self.numruns))
        meas_tmp = self.meas_func()
        if meas_tmp.size ==1:
#            tmp_z = np.zeros((self.meas_size,1))
#            tmp_z[0]=meas_tmp
            if self.init_state_val==0:
                self.x_plus[0,0] = meas_tmp[0,0]+np.random.normal(0,self.sigma)
            else:
                self.x_plus[0,0] = self.init_state_val
        else:
            if self.num_blocks>1:
                self.x_plus[:,0] = np.zeros((self.state_size,))
            else:
                self.x_plus[0,0] = meas_tmp[0,0]

        self.K_cum = np.zeros((self.state_size,self.meas_size,self.numruns))
        self.z = np.zeros((self.meas_size,1,self.numruns))
        self.zhat = np.zeros((self.meas_size,1,self.numruns))
        self.residual =  np.zeros((self.meas_size,1,self.numruns))
        self.exp_residual =  np.zeros((self.meas_size,1,self.numruns))
        self.meas_array = np.random.randn(self.meas_size,1,self.numruns)  # Scalar meas.
        self.k = 0
        self.posgains=[]
        
    def setNumRuns(self,numruns=default_numruns):
        old_numruns = self.numruns
        self.numruns=numruns
        print('Changing numruns from ',old_numruns,' to ',numruns)

    def run(self):
        self.reset()
        rsum=0
        rsumsq = 0
        rexpsum=0
        rexpsumsq = 0
        while self.k<self.numruns-1:
            if self.displayflag:
                self.display()
            self.cycle()
            r = self.residual[0,0,self.k]
            rsum = rsum + r
            rsumsq = rsumsq + r*r
            if self.logmode==1:
                rexp = self.exp_residual[0,0,self.k]
                rexpsum = rexpsum + rexp
                rexpsumsq = rexpsumsq + rexp*rexp
        self.display()
        avg = rsum/self.numruns
        mse = rsumsq/self.numruns
        rms = math.sqrt(mse)
        if self.logmode==1:
            avgexp = rexpsum/self.numruns
            mseexp = rexpsumsq/self.numruns
            rmsexp = math.sqrt(mseexp)
        print('Avg residual: ',avg)
        print('RMS of residual: ', rms)
        if self.logmode==1:
            print('Avg residual of exponentials: ',avgexp)
            print('RMS of residual of exponentials: ', rmsexp)

        return self.x_plus, self.x_minus, self.residual, self.exp_residual

    def cycle(self):
        self.extrapolate()
        self.computeGain()
        self.update()

    def extrapolate(self):
        # Extapolate state and covariance:

        self.k = self.k+1
        curr_dt = self.meas_obj.timestamps[self.k]-self.meas_obj.timestamps[self.k-1]
        if curr_dt>datetime.timedelta(1):
            TempPhi = self.Alt_Phi
            TempQ = self.Alt_Q
            if self.verbose or self.k==self.epoch_dumps:
                print('Taking Alt path.')
                print('Delta Time: ',curr_dt)
        else:
            TempPhi = self.Phi
            TempQ   = self.Q
            if self.verbose or self.k==self.epoch_dumps:
                print('Taking Normal path.')
        if self.verbose or self.k==self.epoch_dumps:
            print('Extrap: Phi @ epoch '+str(self.k)+':')
            print(TempPhi)
            print('Extrap: Prior P is Pos Definite? '+str(ku.is_pos_def(self.P_plus)))
        if self.verbose or self.k==self.epoch_dumps:
            print('Extrap: Prior P: ')
            print(self.P_minus)
            print('Extrap: Prior x: ')
            print(self.x_plus[:,self.k-1])

# State Extrapolation:
#        self.x_minus[:,self.k] = np.dot(self.Phi,self.x_plus[:,self.k-1])
        self.x_minus[:,self.k] = TempPhi @ self.x_plus[:,self.k-1]
        if self.verbose or self.k==self.epoch_dumps:
            print('Extrap: Predicted x: ')
            print(self.x_minus[:,self.k])

# Covariance Extrapoloation:
#        self.P_minus = np.dot(self.Phi,np.dot(self.P_plus,self.Phi.transpose())) + self.Q
        self.P_minus = TempPhi @ self.P_plus @ TempPhi.transpose() + TempQ
# Kludge:  rescaling:
#        self.P_minus = 0.80*self.P_minus
        if self.verbose or self.k==self.epoch_dumps:
            print('Cov Extrap:')
            print(self.P_minus)
#            print('with dot mult:')
#            print(np.dot(TempPhi,np.dot(self.P_plus,TempPhi.transpose())) + TempQ)
        self.P_minus_cum[:,:,self.k] = self.P_minus
        increase = self.P_minus_cum[0,0,self.k]-self.P_plus_cum[0,0,self.k-1]
        if increase>0:
            self.posgains.append(increase)

    def computeGain(self):
        if self.composite:
            self.R = self.meas_obj.genMeasNoiseMatrix()
        S = np.dot(self.H,np.dot(self.P_minus,self.H.transpose()))+self.R
        S_inv = np.linalg.inv(S)
        PH_trans = self.P_minus @ self.H.transpose()
        self.K[:,:] = np.dot(PH_trans,S_inv)
        self.K_cum[:,:,self.k] = self.K
        if self.verbose:
            print('Innovation covariance: ')
            print(S)
            print('S inverse: ')
            print(S_inv)
            print('PH_transpose:')
            print(PH_trans)
            print('Kalman Gain:')
            print(self.K_cum[:,:,self.k])

    def update(self):
        # Update Covariance
        self.P_plus = np.dot((self.I-np.dot(self.K[:,:],self.H)),\
                            np.dot(self.P_minus,\
                                   (self.I-np.dot(self.K[:,:],self.H)).transpose())) \
                     + np.dot(self.K[:,:],np.dot(self.R,self.K[:,:].transpose()))
        self.P_plus_cum[:,:,self.k] = self.P_plus
        if self.verbose:
            print('Update: Pos Definite? '+str(ku.is_pos_def(self.P_plus)))
            print('Update: Corrected P: ')
            print(self.P_plus)
        meas_tmp = self.meas_func()
#        print('meas_tmp shape: ' + str(meas_tmp.shape))
        if self.num_blocks>1:
            self.z[:,0,self.k] = meas_tmp.reshape((self.num_blocks-1,))
        else:
            self.z[:,0,self.k] = meas_tmp.reshape((self.meas_size,))
        self.zhat[:,0,self.k] = np.dot(self.H,self.x_minus[:,self.k])
        self.residual[:,0,self.k] = self.z[:,0,self.k] - self.zhat[:,0,self.k]
        self.exp_residual[:,0,self.k] = np.exp(self.z[:,0,self.k]) - np.exp(self.zhat[:,0,self.k])
        weighted_residual = np.dot(self.K[:,:],self.residual[:,0,self.k])
        self.x_plus[:,self.k] = self.x_minus[:,self.k] + weighted_residual
        if self.verbose:
            print('Update: Measurement')
            print(self.z[:,0,self.k])
            print('Update:  Corrected state: ')
            print(self.x_plus[:,self.k])
            print('Update: Residual: ')
            print(self.residual[:,0,self.k])

#        new_state = self.x_plus[:,self.k]
#        print(new_state)
        

    def display(self):
        if self.k==0:
            print('\nStart\n')
            print('*************Initialization*******************')
            print('*************Filter ID: '+self.filter_id+'*******************')
            print('k: ',self.k)
            print('Prior State: (N/A)\n',self.x_minus[:,self.k])
            print('Prior Cov:\n', self.P_minus)
            print('Kalman Gain: (N/A)\n',self.K[:,:])
            print('Measurement: (N/A)\n',self.z[:,self.k])
            print('Pre-fit Residual:(N/A)\n',self.residual[:,self.k])
            print('Initial Posterior State:\n',self.x_plus[:,self.k])
            print('Initial Posterior Cov:\n', self.P_plus)
            print('**************End Initialization***************')
        else:            
            print('**********************************************')
            print('*************Filter ID: '+self.filter_id+'*******************')
            print('k: ',self.k)
            print('Prior State:\n',self.x_minus[:,self.k])
            print('Prior Cov:\n', self.P_minus)
            print('Kalman Gain:\n',self.K[:,:])
            print('Measurement:\n',self.z[:,0,self.k])
            if self.logmode==1:
                print('Residual of Exponentials:\n',self.exp_residual[:,0,self.k])
            if self.x_minus[0,self.k] != 0.0:
                print('Residual as percent of State:\n',(self.residual[:,0,self.k]/self.x_minus[0,self.k])*100,'%')
            print('Posterior State:\n',self.x_plus[:,self.k])
            print('Posterior Cov:\n', self.P_plus)
            print('**********************************************')


    def test_meas_func(self):
        return self.meas_array[(0,self.k)]

    def dump(self):
        print(self.filter_id,"Model Dump:")
        print("Process Model:")
        print("Phi:  Shape: ",self.Phi.shape)
        print(self.Phi)
        print("Alt_Phi: Shape: ",self.Alt_Phi.shape,'\n',self.Alt_Phi)
        print("Q: Shape: ",self.Q.shape,'\n',self.Q,'\nAlt_Q:\n',self.Alt_Q)
        print("Measurement Model: ")
        print("H: Shape: ",self.H.shape,'\n')
        print(self.H)
        print("R: Shape: ",self.R.shape)
        if ku.is_pos_def(self.R):
            print('pos definite')
        else:
            print('not pos definite')
        print(self.R)
        print('Initial Covariance: Shape: ',self.P_minus.shape)
        print(self.P_minus)
        print('Initial Gain: Shape: ',self.K.shape)
        print(self.K)

