import kalman_filter as kf
import stockmeas as sm
import kalplots as kp
import numpy as np
import json

# This filter is not a composite, but instead builds a weighted average of each filter's output
# This would be a good opportunity to use class variables and static functions.  The compute the averages
# after each cycle.  But that will be a later enhancement.
# IBM stats:
# mean:	0.00004333	-0.00000029
# variance:	0.00027046	0.00056898
# exp(mean)	1.00004333	0.99999971
# exp(variance)	1.00027050	1.00056914
mynumruns =20
stocks = ['IBM','AMZN','GE','MSFT','IBM','AMZN','GE','MSFT','IBM','AMZN','GE','MSFT','IBM','AMZN','GE','MSFT']
q_factors = [.1,.1,.1,.1,.2,.2,.2,.2,.3,.3,.3,.3,.4,.4,.4,.4]
mykfs = []
i=0
for stock in stocks:
    infiles=['daily_adjusted_'+stock+'.csv']
    title_prefix='Std 3-state filter: '+stock
    my_legend_str = [stock]
    mysm_tmp = sm.StockMeasurement(noiseSigma=0, # added measurement noise
                            logmode=1,
                            infiles=infiles
                            )
    q_factor = q_factors[i]
    i = i+1
    mykf_tmp =kf.KalmanFilter(meas_obj=mysm_tmp,
                        meas_func=mysm_tmp.nextMeas, 
                        basic_state_size=3,
                        meas_size = 1,
                        dt=1,  # what unit of time?  Daily (1)?, seconds (24*3600)?
                        phi_type=1,
                        sigma=0.000001,  # Stock prices are accurately reported
                        num_runs=mynumruns,
                        logmode=1, 
                        num_blocks=1,
                        displayflag=False,
                        verbose=False,
                        q_factor=q_factor)

    # mykf_tmp.Basic_Q = q_factor * np.array([[0,0,0],
    #                           [0, 0.00027, 0],
    #                           [0,0,0.000568]])
    # mykf_tmp.Q = mykf_tmp.Basic_Q        
    # mykf_tmp.Alt_Q = 9*mykf_tmp.Q
    mykf_tmp.run()
    mykfs.append(mykf_tmp)

# Create and use a KF object, but don't actually run it. 
title_prefix='Std 3-state filter: Weighted Average '
my_legend_str = ['Weighted Average']
# This is just a dummy
mysm = sm.StockMeasurement(noiseSigma=0, # added measurement noise
                           logmode=0,  # measurements already in logmode from subfilters
                           infiles=None,
                           subfilters=True,
                           subfilter_list=mykfs
                          )
#                           infiles=['daily_AMZN_stockmeas.csv',
#                                    'GE-2000-Aug-1_to_2015-September-04.csv'])
basic_size=3
mykf = kf.KalmanFilter(meas_obj=mysm,
                       meas_func=mysm.nextMeas, 
                       basic_state_size=basic_size,
                       meas_size = len(stocks)*basic_size,
                       dt=1,  # what unit of time?  Daily (1)?, seconds (24*3600)?
                       phi_type=1,
                       sigma=0.0000001,  # Stock prices are accurately reported
                       num_runs=mynumruns,
                       logmode=1, 
                       num_blocks=1,
                       composite=False,
                       displayflag=False,
                       verbose=False)

sum_of_recip = np.zeros((basic_size, 1))
weights = np.zeros((len(stocks), basic_size))
recip_array = np.zeros((len(stocks),basic_size))
for i in range(mynumruns):
    weighted_sum = np.zeros((basic_size,1))
    for j in range(len(stocks)):
        recip_array[j,:] = 1/np.diag(mykfs[j].P_plus_cum[:,:,i])
        sum_of_recip += recip_array[j,:].reshape((basic_size,1))
    for j in range(len(stocks)):
        numer = recip_array[j,:].reshape((basic_size,1))
        weights[j,:] = (numer/sum_of_recip).reshape((basic_size,))
        summand = mykfs[j].x_plus[:,i].reshape((basic_size,1))
        weight = weights.transpose()[:,j].reshape((basic_size,1))
        summand = weight * summand
        weighted_sum +=  summand
    mykf.x_plus[:,i] = weighted_sum.reshape((3,))

# kp.std_sawtooth_plot(fignum=1,kfobj=mykf,expflag=1, 
#                      last_percent=1,
#                      title_prefix=title_prefix)
# kp.plot_residuals(kfobj=mykf,expflag=1,
#                   title_prefix=title_prefix,
#                   legend_str=my_legend_str)
#kp.plot_posgains(kfobj=mykf,expflag=1)
#kp.plot_gains(kfobj=mykf,state=0)
kp.plot_states(kfobj=mykf)


