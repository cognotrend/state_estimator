import kalman_filter as kf
import stockmeas as sm
mysm = sm.StockMeasurement(noiseSigma=5)
mykf = kf.KalmanFilter(meas_func=mysm.nextMeas)
def do_it():
    return mykf


