import numpy as np
import csv
import datetime as dt
import pandas as pd
import random
''' Resources:
1)  "Lognormal Model for Stock Prices", Michael J. Sharpe, UCD Math Dept, 
    http://www.math.ucsd.edu/~msharpe/stockgrowth.pdf
TODO:
    1) Add processing of multiple stocks if num stocks >1  Done
    2) Ensure default of a single direct measurement if num stocks =1
'''
class StockMeasurement():
    def __init__(self,
                 noiseSigma=0,
                 logmode=0, 
                 infiles=['daily_adjusted_IBM.csv'],
                 subfilters=False,
                 subfilter_list = []
                 ):
        self.subfilters = subfilters
        self.subfilters_list = subfilter_list
        if self.subfilters==False:
            self.num_stocks = len(infiles)
            self.logmode = logmode
            self.noiseSigma = noiseSigma
            d_parser = lambda x: dt.datetime.fromisoformat(x)

            
            self.dfs = []
            for stock_file in infiles:
                df = pd.read_csv(stock_file, 
                                index_col='timestamp', 
                                parse_dates=['timestamp'],
                                date_parser=d_parser)
                # Reverse data coming from Alpha Vantage
                df = df[::-1]
                self.dfs.append(df)
                
            self.timestamps = self.dfs[0].index
            
            self.myiter = iter(self.timestamps)
                
            self.meas_func = self.nextMeas
            self.meas_array = np.zeros((self.num_stocks-1,1))
        else:
            self.next_ts_index = 0
            self.logmode = logmode
            self.noiseSigma = noiseSigma

            timestamps_list = list(range(0,self.subfilters_list[0].numruns))
            self.ts_indices = list(range(0,self.subfilters_list[0].numruns))
            self.timestamps = []
            curr_date = dt.datetime(year=2020,month=1,day=1)
            for ts in timestamps_list:
                self.timestamps.append(curr_date)
                curr_date = curr_date + dt.timedelta(1)
            self.myiter = iter(self.timestamps)
            self.ts_index_iter = iter(self.ts_indices)
            
            self.ms = []
            self.Ps = []
            self.meas_size = 0
            for kf in self.subfilters_list:
                m = kf.x_plus
                P = kf.P_plus_cum 
                self.ms.append(m)
                self.Ps.append(P)
                self.meas_size += kf.state_size
#            self.meas_size *= len(self.ms)                
            self.meas_func = self.nextMeas
            self.meas_array = np.zeros((self.meas_size,1))  # meas vector
      

    def nextMeas(self):
        next_tstamp = next(self.myiter)

        if self.subfilters==False:
            if self.logmode==1:
                # hard coded for ref=0
                ref_meas = np.log(self.dfs[0].loc[next_tstamp]['open'])
            else:
                ref_meas = self.dfs[0].loc[next_tstamp]['open']
            if self.num_stocks>1:
                for i in list(range(0,self.num_stocks-1)):
                    if self.logmode==1:
                        self.meas_array[i]= np.log(self.dfs[i+1].loc[next_tstamp]['open']) - ref_meas
                    else:
                        self.meas_array[i]= self.dfs[i+1].loc[next_tstamp]['open'] - ref_meas
                noise_array = np.random.normal(0,self.noiseSigma,(self.num_stocks-1,1))
        #        print('meas_array shape: '+str(self.meas_array.shape))
        #        print('noise_array shape: '+ str(noise_array.shape))
                meas = self.meas_array[:,0] + noise_array[:,0]
                meas = meas.reshape((self.num_stocks-1,1))
        #        print('meas shape: '+str(meas.shape))
            else:
                meas = (ref_meas+random.gauss(0, self.noiseSigma))*np.ones((1,1))
            return meas
        else:
            self.next_ts_index = next(self.ts_index_iter)
            tmp = []
            tmp_noise = []
            for i in range(0,len(self.ms)):
                tmp.append(self.ms[i][:,self.next_ts_index])
                tmpPdiag = 0*np.diag(self.Ps[i][:,:,self.next_ts_index]) # temp zero!
                Psize = tmpPdiag.size
                tmpNoiseArray = np.zeros((Psize,1))
                for j in list(range(0,Psize)):
                    tmpNoiseArray[j,0]= np.random.normal(0,tmpPdiag[j])
                tmp_noise.append(tmpNoiseArray)
            self.meas_array = np.concatenate(tmp,axis=0).reshape((self.meas_size,1)) 
            if self.logmode==1:
                self.meas_array = np.log(self.meas_array)
            noise_array = np.concatenate(tmp_noise,axis=0)
            meas = self.meas_array[:,0].reshape((self.meas_size,1)) +  noise_array
            meas = meas.reshape((self.meas_size,1))
        return meas

    def genMeasNoiseMatrix(self):
        if self.subfilters:
            R = np.zeros((self.meas_size,self.meas_size))
            numblocks = len(self.ms)
            (blocksize,dontcare) = self.Ps[0][:,:,self.next_ts_index].shape
            for i in list(range(0,numblocks)):
                R[i*blocksize:(i+1)*blocksize,i*blocksize:(i+1)*blocksize]=self.Ps[0][:,:,self.next_ts_index]
            return R
        else:
            return None



    def reset(self):
        self.index = -1

    def setNoise(self, sigma=0):
        self.noiseSigma = sigma
