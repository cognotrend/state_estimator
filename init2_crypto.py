import kalman_filter as kf
import stockmeas as sm
import kalplots as kp
import numpy as np
import json


print('\n\n******','Start: init2_crypto.py','******\n\n')
infiles=['currency_daily_BTC_USD.csv']
#,
#         'daily_adjusted_AMZN.csv',
#         'daily_adjusted_GE.csv',
#         'daily_adjusted_MSFT.csv']
title_prefix='Std 3-state filter: BTC '

my_legend_str = ['BTC']
#my_legend_str = ['Amzn','GE','MSFT']
mysm = sm.StockMeasurement(noiseSigma=0, # added measurement noise
                           logmode=0,
                           infiles=infiles
                          )

mykf = kf.KalmanFilter(meas_obj=mysm,
                       meas_func=mysm.nextMeas, 
                       basic_state_size=3,
                       init_state_val = 7000.0,
                       meas_size = 1,
                       dt=1,  # what unit of time?  Daily (1)?, seconds (24*3600)?
                       phi_type=1,
                       sigma=1.0,  # Crypto prices are accurately reported
                       sigma1=6.0,
                       sigma2=4.0,
                       sigma3=1.0,
                       num_runs=-1,
                       logmode=0, 
                       num_blocks=1,
                       displayflag=False,
                       verbose=False)

mykf.Basic_Q = np.array([[10,0,0],
                          [0, 4, 0],
                          [0,0,3.0]])
mykf.Q = mykf.Basic_Q        
mykf.Alt_Q = mykf.Q

mykf.run()
'''
state=2
v1=mykf.P_minus_cum[state,state,:].reshape(mykf.numruns,)
v2=mykf.P_plus_cum[state,stat[e,:].reshape(mykf.numruns,)
doubletime=kp.collate(np.arange(0,mykf.numruns),np.arange(0,mykf.numruns))
sawtooth=kp.collate(v1,v2)
plt.figure(1)
plt.plot(doubletime,sawtooth)
plt.show()
'''
kp.std_sawtooth_plot(fignum=1,kfobj=mykf,
                    expflag=0, 
                    last_percent=1,
                    title_prefix=title_prefix)
kp.plot_residuals(kfobj=mykf,
                expflag=0,
                  title_prefix=title_prefix,
                  legend_str=my_legend_str)

kp.plot_states(kfobj=mykf, expflag=0)


kp.plot_gains(kfobj=mykf,state=0)

mykf.dump()

