import kalman_filter as kf
import stockmeas as sm
import kalplots as kp
import numpy as np
import json

infiles=['daily_adjusted_IBM.csv']
#,
#         'daily_adjusted_AMZN.csv',
#         'daily_adjusted_GE.csv',
#         'daily_adjusted_MSFT.csv']
title_prefix='Std 3-state filter: IBM '
# IBM
# mean:	0.00004333	-0.00000029
# variance:	0.00027046	0.00056898
# exp(mean)	1.00004333	0.99999971
# exp(variance)	1.00027050	1.00056914

my_legend_str = ['IBM']
#my_legend_str = ['Amzn','GE','MSFT']
mysm = sm.StockMeasurement(noiseSigma=0, # added measurement noise
                           logmode=1,
                           infiles=infiles
                          )
#                           infiles=['daily_AMZN_stockmeas.csv',
#                                    'GE-2000-Aug-1_to_2015-September-04.csv'])

mykf = kf.KalmanFilter(meas_obj=mysm,
                       meas_func=mysm.nextMeas, 
                       basic_state_size=3,
                       meas_size = 1,
                       dt=1,  # what unit of time?  Daily (1)?, seconds (24*3600)?
                       phi_type=1,
                       sigma=0.000001,  # Stock prices are accurately reported
                       num_runs=35,
                       logmode=1, 
                       num_blocks=1,
                       displayflag=False,
                       verbose=False)

mykf.Basic_Q = np.array([[1,0,0],
                          [0, 0.00027, 0],
                          [0,0,0.000568]])
mykf.Q = mykf.Basic_Q        
mykf.Alt_Q = 9*mykf.Q
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
kp.std_sawtooth_plot(fignum=1,kfobj=mykf,expflag=1, 
                     last_percent=1,
                     title_prefix=title_prefix)
kp.plot_residuals(kfobj=mykf,expflag=1,
                  title_prefix=title_prefix,
                  legend_str=my_legend_str)
#kp.plot_posgains(kfobj=mykf,expflag=1)
kp.plot_gains(kfobj=mykf,state=0)

mykf.dump()

