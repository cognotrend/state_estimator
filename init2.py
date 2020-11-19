import kalman_filter as kf
import stockmeas as sm
import kalplots as kp
#myinfile = 'GE-2000-Aug-1_to_2015-September-04.csv'
#myinfile = 'daily_AMZN_stockmeas.csv'
#myinfile = 'ratio_amsn_ge.csv'
title_prefix='AMZN: '
mysm = sm.StockMeasurement(noiseSigma=0, # added measurement noise
                           logmode=1,
                           infiles=['daily_AMZN_stockmeas.csv']
                           )
mykf = kf.KalmanFilter(meas_obj=mysm,
                       meas_func=mysm.nextMeas, 
                       state_size=3, 
                       dt=1,  # what unit of time?  Daily (1)?, seconds (24*3600)?
                       phi_type=1,
                       sigma=0.01,  # Stock prices are accurately reported
                       num_runs=200,
                       logmode=1, 
                       num_blocks=1,
                       displayflag=False)
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
kp.std_sawtooth_plot(fignum=1,kfobj=mykf,expflag=1, 
                     last_percent=1,
                     title_prefix=title_prefix)
kp.plot_residuals(kfobj=mykf,expflag=1,
                  title_prefix=title_prefix)
#kp.plot_posgains(kfobj=mykf,expflag=1)