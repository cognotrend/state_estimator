import kalman_filter as kf
import stockmeas as sm
mysm = sm.StockMeasurement(noiseSigma=0)
mykf = kf.KalmanFilter(meas_func=mysm.nextMeas, state_size=3, phi_type=1)
def do_it():
    return mykf


