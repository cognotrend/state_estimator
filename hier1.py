import kalman_filter as kf
import stockmeas as sm
import kalplots as kp
import numpy as np
import json

# IBM
# mean:	0.00004333	-0.00000029
# variance:	0.00027046	0.00056898
# exp(mean)	1.00004333	0.99999971
# exp(variance)	1.00027050	1.00056914
mynumruns =1000
infiles=['daily_adjusted_IBM.csv']
#,
#         'daily_adjusted_AMZN.csv',
#         'daily_adjusted_GE.csv',
#         'daily_adjusted_MSFT.csv']
title_prefix='Std 3-state filter: IBM '
my_legend_str = ['IBM']
#my_legend_str = ['Amzn','GE','MSFT']
mysm1 = sm.StockMeasurement(noiseSigma=0, # added measurement noise
                           logmode=1,
                           infiles=infiles
                          )
#                           infiles=['daily_AMZN_stockmeas.csv',
#                                    'GE-2000-Aug-1_to_2015-September-04.csv'])

mykf1 = kf.KalmanFilter(meas_obj=mysm1,
                       meas_func=mysm1.nextMeas, 
                       basic_state_size=3,
                       meas_size = 1,
                       dt=1,  # what unit of time?  Daily (1)?, seconds (24*3600)?
                       phi_type=1,
                       sigma=0.000001,  # Stock prices are accurately reported
                       num_runs=mynumruns,
                       logmode=1, 
                       num_blocks=1,
                       displayflag=False,
                       verbose=False)

mykf1.Basic_Q = np.array([[1,0,0],
                          [0, 0.00027, 0],
                          [0,0,0.000568]])
mykf1.Q = mykf1.Basic_Q        
mykf1.Alt_Q = 9*mykf1.Q
mykf1.run()

infiles=['daily_adjusted_AMZN.csv']
#,
#         'daily_adjusted_AMZN.csv',
#         'daily_adjusted_GE.csv',
#         'daily_adjusted_MSFT.csv']
title_prefix='Std 3-state filter: AMZN '
my_legend_str = ['AMZN']
#my_legend_str = ['Amzn','GE','MSFT']
mysm2 = sm.StockMeasurement(noiseSigma=0, # added measurement noise
                           logmode=1,
                           infiles=infiles
                          )
#                           infiles=['daily_AMZN_stockmeas.csv',
#                                    'GE-2000-Aug-1_to_2015-September-04.csv'])

mykf2 = kf.KalmanFilter(meas_obj=mysm2,
                       meas_func=mysm2.nextMeas, 
                       basic_state_size=3,
                       meas_size = 1,
                       dt=1,  # what unit of time?  Daily (1)?, seconds (24*3600)?
                       phi_type=1,
                       sigma=0.000001,  # Stock prices are accurately reported
                       num_runs=mynumruns,
                       logmode=1, 
                       num_blocks=1,
                       displayflag=False,
                       verbose=False)

mykf2.Basic_Q = np.array([[1,0,0],
                          [0, 0.00027, 0],
                          [0,0,0.000568]])
mykf2.Q = mykf2.Basic_Q        
mykf2.Alt_Q = 9*mykf2.Q
mykf2.run()


infiles=['daily_adjusted_GE.csv']
#,
#         'daily_adjusted_AMZN.csv',
#         'daily_adjusted_GE.csv',
#         'daily_adjusted_MSFT.csv']
title_prefix='Std 3-state filter: GE '
my_legend_str = ['GE']
#my_legend_str = ['Amzn','GE','MSFT']
mysm3 = sm.StockMeasurement(noiseSigma=0, # added measurement noise
                           logmode=1,
                           infiles=infiles
                          )
#                           infiles=['daily_AMZN_stockmeas.csv',
#                                    'GE-2000-Aug-1_to_2015-September-04.csv'])

mykf3 = kf.KalmanFilter(meas_obj=mysm3,
                       meas_func=mysm3.nextMeas, 
                       basic_state_size=3,
                       meas_size = 1,
                       dt=1,  # what unit of time?  Daily (1)?, seconds (24*3600)?
                       phi_type=1,
                       sigma=0.000001,  # Stock prices are accurately reported
                       num_runs=mynumruns,
                       logmode=1, 
                       num_blocks=1,
                       displayflag=False,
                       verbose=False)

mykf3.Basic_Q = np.array([[1,0,0],
                          [0, 0.00027, 0],
                          [0,0,0.000568]])
mykf3.Q = mykf3.Basic_Q        
mykf3.Alt_Q = 9*mykf3.Q
mykf3.run()

infiles=['daily_adjusted_MSFT.csv']
#,
#         'daily_adjusted_AMZN.csv',
#         'daily_adjusted_GE.csv',
#         'daily_adjusted_MSFT.csv']
title_prefix='Std 3-state filter: MSFT '
my_legend_str = ['MSFT']
#my_legend_str = ['Amzn','GE','MSFT']
mysm4 = sm.StockMeasurement(noiseSigma=0, # added measurement noise
                           logmode=1,
                           infiles=infiles
                          )
#                           infiles=['daily_AMZN_stockmeas.csv',
#                                    'GE-2000-Aug-1_to_2015-September-04.csv'])

mykf4 = kf.KalmanFilter(meas_obj=mysm4,
                       meas_func=mysm4.nextMeas, 
                       basic_state_size=3,
                       meas_size = 1,
                       dt=1,  # what unit of time?  Daily (1)?, seconds (24*3600)?
                       phi_type=1,
                       sigma=0.000001,  # Stock prices are accurately reported
                       num_runs=mynumruns,
                       logmode=1, 
                       num_blocks=1,
                       displayflag=False,
                       verbose=False)

mykf4.Basic_Q = np.array([[1,0,0],
                          [0, 0.00027, 0],
                          [0,0,0.000568]])
mykf4.Q = mykf4.Basic_Q        
mykf4.Alt_Q = 9*mykf4.Q
mykf4.run()

infiles=['daily_adjusted_MSFT.csv']
#,
#         'daily_adjusted_AMZN.csv',
#         'daily_adjusted_GE.csv',
#         'daily_adjusted_MSFT.csv']
title_prefix='Std 3-state filter: Composite '
my_legend_str = ['Composite']
#my_legend_str = ['Amzn','GE','MSFT']
mysm = sm.StockMeasurement(noiseSigma=0, # added measurement noise
                           logmode=0,  # measurements already in logmode from subfilters
                           infiles=None,
                           subfilters=True,
                           subfilter_list=[mykf1,mykf2,mykf2,mykf4]
                          )
#                           infiles=['daily_AMZN_stockmeas.csv',
#                                    'GE-2000-Aug-1_to_2015-September-04.csv'])

mykf = kf.KalmanFilter(meas_obj=mysm,
                       meas_func=mysm.nextMeas, 
                       basic_state_size=3,
                       meas_size = 12,
                       dt=1,  # what unit of time?  Daily (1)?, seconds (24*3600)?
                       phi_type=1,
                       sigma=0.0000001,  # Stock prices are accurately reported
                       num_runs=mynumruns,
                       logmode=1, 
                       num_blocks=1,
                       composite=True,
                       displayflag=False,
                       verbose=False)

mykf.Basic_Q = np.array([[0.1,0,0],
                          [0, 0.000027, 0],
                          [0,0,0.0000568]])
mykf.Q = mykf.Basic_Q        
mykf.Alt_Q = 9*mykf.Q
mykf.run()
mykf.dump()


kp.std_sawtooth_plot(fignum=1,kfobj=mykf,expflag=1, 
                     last_percent=1,
                     title_prefix=title_prefix)
kp.plot_residuals(kfobj=mykf,expflag=1,
                  title_prefix=title_prefix,
                  legend_str=my_legend_str)
#kp.plot_posgains(kfobj=mykf,expflag=1)
kp.plot_gains(kfobj=mykf,state=0)


