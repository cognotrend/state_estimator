import kalman_filter as kf
import stockmeas as sm
import kalplots as kp
infiles=['daily_adjusted_IBM.csv',
         'daily_adjusted_AMZN.csv',
         'daily_adjusted_GE.csv',
         'daily_adjusted_MSFT.csv']
title_prefix='Price Diff Filter IBM=reference: '
mysm = sm.StockMeasurement(noiseSigma=0, # added measurement noise
                           logmode=1,
                           infiles=infiles
                          )
#                           infiles=['daily_AMZN_stockmeas.csv',
#                                    'GE-2000-Aug-1_to_2015-September-04.csv'])

mykf = kf.KalmanFilter(meas_obj=mysm,
                       meas_func=mysm.nextMeas, 
                       basic_state_size=3,
                       meas_size = 3,
                       dt=1/3,  # what unit of time?  Daily (1)?, seconds (24*3600)?
                       phi_type=1,
                       sigma=0.000001,  # Stock prices are accurately reported
                       num_runs=50,
                       logmode=1, 
                       num_blocks=4,
                       displayflag=True,
                       verbose=False)

mykf.dump()
print('Resetting:')
mykf.reset()
print('After reset:')
mykf.dump()
