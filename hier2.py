import kalman_filter as kf
import stockmeas as sm
import kalplots as kp
import numpy as np
import json
import argparse
# This filter is not a composite, but instead builds a weighted average of each 
# filter's output.
# This would be a good opportunity to use class variables and static functions.  
# The compute the averages after each cycle.  But that will be a later enhancement.
# TODO:
# Improve argument handling with argparse
# IBM stats:
# mean:	0.00004333	-0.00000029
# variance:	0.00027046	0.00056898
# exp(mean)	1.00004333	0.99999971
# exp(variance)	1.00027050	1.00056914
parser = argparse.ArgumentParser()
parser.add_argument('--numruns','-nr', help='Number of kalman cycles to execute.')
parser.add_argument('--stocks','-s', help='Comma-separated list of stocks.')
parser.add_argument('--lmode','-lm',help='Run in log mode if present.')
args = parser.parse_args()
if args.numruns is not None:
    mynumruns = int(args.numruns)
else:
    mynumruns = 200

if args.numruns is not None:
    stock_str = args.stocks
    if stock_str.find(',')>-1:
        stocks = stock_str.split(',')
    else:
        stocks = [stock_str]
# stocks = ['IBM','AMZN','GE','MSFT','IBM','AMZN','GE','MSFT','IBM','AMZN','GE','MSFT','IBM','AMZN','GE','MSFT']
# q_factors = [.1,1.0,10.0,100.0,.2,.2,.2,.2,.3,.3,.3,.3,.4,.4,.4,.4]
else:
    stocks = ['IBM','GE','MSFT']

if args.lmode is not None:
    logmode = 1
else:
    logmode = 0

q_factors = [1.0,1.0,1.0]
mykfs = []
i=0
for stock in stocks:
    infiles=['daily_adjusted_'+stock+'.csv']
    title_prefix='Std 3-state filter: '+stock
    my_legend_str = [stock]
    mysm_tmp = sm.StockMeasurement(noiseSigma=10, # added measurement noise
                            logmode=logmode,
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
                        sigma=5.0,  # BTC prices are accurately reported
                        num_runs=mynumruns,
                        logmode=logmode, 
                        num_blocks=1,
                        displayflag=False,
                        verbose=False,
                        q_factor=q_factor,
                        filter_id=stock,
                        epoch_dumps=-1)

    # mykf_tmp.Basic_Q = q_factor * np.array([[0,0,0],
    #                           [0, 0.00027, 0],
    #                           [0,0,0.000568]])
    # mykf_tmp.Q = mykf_tmp.Basic_Q        
    # mykf_tmp.Alt_Q = 9*mykf_tmp.Q
    mykf_tmp.run()
    kp.std_sawtooth_plot(fignum=1,kfobj=mykf_tmp,expflag=0, 
                      last_percent=1,
                      title_prefix=title_prefix)
    kp.plot_residuals(kfobj=mykf_tmp,expflag=0,
                   title_prefix=title_prefix,
                   legend_str=my_legend_str)
      
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
                       meas_size = len(stocks),
                       dt=1,  # what unit of time?  Daily (1)?, seconds (24*3600)?
                       phi_type=1,
                       sigma=0.01,  # Stock prices are accurately reported
                       num_runs=mynumruns,
                       logmode=1, 
                       num_blocks=1,
                       composite=False,
                       displayflag=False,
                       verbose=False,
                       filter_id='Composite')

# Since mykf.run() is not called, the following serves to compute all the values of 
# mykf.x_minus, x_plus, z, zhat.  P_minus_cum, etc. are not actually computed.
for i in range(mynumruns):
    sum_of_recip = np.zeros((basic_size, 1))
    sum_of_recip_minus = np.zeros((basic_size, 1))
    weights = np.zeros((len(stocks), basic_size))
    weights_minus = np.zeros((len(stocks), basic_size))
    recip_array = np.zeros((len(stocks),basic_size))
    recip_array_minus = np.zeros((len(stocks),basic_size))
    weighted_sum = np.zeros((basic_size,1))
    weighted_sum_minus = np.zeros((basic_size,1))
    for j in range(len(stocks)):
        recip_array[j,:] = 1/np.diag(mykfs[j].P_plus_cum[:,:,i])
        recip_array_minus[j,:] = 1/np.diag(mykfs[j].P_minus_cum[:,:,i])
        sum_of_recip += recip_array[j,:].reshape((basic_size,1))
        sum_of_recip_minus += recip_array[j,:].reshape((basic_size,1))
    for j in range(len(stocks)):
        numer = recip_array[j,:].reshape((basic_size,1))
        numer_minus = recip_array_minus[j,:].reshape((basic_size,1))
        weights[j,:] = (numer/sum_of_recip).reshape((basic_size,))
        weights_minus[j,:] = (numer_minus/sum_of_recip_minus).reshape((basic_size,))
        summand = mykfs[j].x_plus[:,i].reshape((basic_size,1))
        summand_minus = mykfs[j].x_minus[:,i].reshape((basic_size,1))
        weight = weights.transpose()[:,j].reshape((basic_size,1))
        weight_minus = weights_minus.transpose()[:,j].reshape((basic_size,1))
        summand = weight * summand
        summand_minus = weight_minus * summand_minus
        weighted_sum +=  summand
        weighted_sum_minus += summand_minus
        mykf.z[j,0,i] = mykfs[j].z[0,0,i]

    mykf.x_plus[:,i] = weighted_sum.reshape((3,))
    mykf.x_minus[:,i] = weighted_sum_minus.reshape((3,))
    tmp = mykf.x_minus[:,i].reshape((3,1))
    if mykf.z[:,:,i].shape != (1,1):
        tmpz = mykf.z[:,:,i].reshape((3,1))
        mykf.H[1,0]=1.0
        mykf.H[2,0]=1.0
    else:
        tmpz = mykf.z[:,:,i]
    tmp_zhat = np.dot(mykf.H,tmp)
    mykf.residual[:,:,i] = tmpz - tmp_zhat

# Not an accurate sawtooth, since P_minus_cum, P_plus_cum are not yet set.
# Need to verify this using debugger.
kp.std_sawtooth_plot(fignum=1,kfobj=mykf,expflag=0, 
                      last_percent=1,
                      title_prefix=title_prefix)
mykf.exp_residual = np.exp(mykf.residual)
kp.plot_residuals(kfobj=mykf,expflag=0,
                   title_prefix=title_prefix,
                   legend_str=my_legend_str)
#kp.plot_posgains(kfobj=mykf,expflag=1)
#kp.plot_gains(kfobj=mykf,state=0)
kp.plot_states(kfobj=mykf)

print("\n*************\nEnd simulations\n******************************************\n******************************************\n")


