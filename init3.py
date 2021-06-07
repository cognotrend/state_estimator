'''
    File name: init3.py
    Author: Michael Carroll
    Date created: 12/06/2020
    Date last modified: 6/07/2021
    Python Version: 3.7.6
    Description:  Test Driver for stockmeas and KalmanFilter objects.
    4-stock scenario.
    Current Issues:  
    2021-06-07:  Works with stocks.  Also works with four copies of BTC data file, but the .csv file
    downloaded from alphavantage had to wrangled:  1)  Redundant USD columns removed and 2) Had to remove '  (USD)' from
    open, close, etc. column headers.
    AttributeError: NoneType object has no attribute timestamps
    TODO:
    Make clone to work with crypto currency
    Create tool to modify .csv files to put cryptos into needed format.                                                                                                      
'''
import kalman_filter as kf
import stockmeas as sm
import kalplots as kp
#infiles=['daily_adjusted_IBM.csv',
#         'daily_adjusted_AMZN.csv',
#         'daily_adjusted_GE.csv',
#         'daily_adjusted_MSFT.csv']
infiles=['currency_daily_BTC_USD.csv',
         'currency_daily_BTC_USD.csv',
         'currency_daily_BTC_USD.csv',
         'currency_daily_BTC_USD.csv']
title_prefix='Price Diff Filter BTC=reference: '
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
