import kalman_filter as kf
import stockmeas as sm
mysm = sm.StockMeasurement(noiseSigma=0.0001)
mykf = kf.KalmanFilter(meas_func=mysm.nextMeas)
mykf.run()

def do_it():
    return mykf


