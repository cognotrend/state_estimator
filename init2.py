import kalman_filter as kf
import stockmeas as sm
import kalplots as kp
#import matplotlib.pyplot as plt
#import numpy as np
# 'GE-2000-Aug-1_to_2015-September-04.csv'
# 'daily_AMZN_stockmeas.csv'
mysm = sm.StockMeasurement(noiseSigma=0,
                           logmode=1,
                           infile='daily_AMZN_stockmeas.csv')
mykf = kf.KalmanFilter(meas_func=mysm.nextMeas, state_size=3, 
                       logmode=1, sigma=1,
                       num_runs=40,phi_type=1,
                       num_blocks=4)
mykf.run()
'''
state=2
v1=mykf.P_minus_cum[state,state,:].reshape(mykf.numruns,)
v2=mykf.P_plus_cum[state,state,:].reshape(mykf.numruns,)
doubletime=kp.collate(np.arange(0,mykf.numruns),np.arange(0,mykf.numruns))
sawtooth=kp.collate(v1,v2)
plt.figure(1)
plt.plot(doubletime,sawtooth)
plt.show()
'''
kp.std_sawtooth_plot(fignum=1,kfobj=mykf)
