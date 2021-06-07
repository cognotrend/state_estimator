'''
    File name: init1.py
    Author: Michael Carroll
    Date created: 7/20/2017
    Date last modified: 6/07/2021
    Python Version: 3.7.6
    Description:  Test Driver for stockmeas and KalmanFilter objects.
    Current Issues:  
    2021-06-07:  
    File kalman_filter.py, line 255, in extrapolate
    curr_dt = self.meas_obj.timestamps[self.k]-self.meas_obj.timestamps[self.k-1]
    AttributeError: NoneType object has no attribute timestamps
    TODO:
    Make clone to work with crypto currency
'''
import kalman_filter as kf
import stockmeas as sm
mysm = sm.StockMeasurement(noiseSigma=0.0001)
mykf = kf.KalmanFilter(meas_func=mysm.nextMeas)
mykf.run()

def do_it():
    return mykf


