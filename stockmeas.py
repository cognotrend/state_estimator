import numpy as np
import csv
import datetime as dt
import pandas as pd

''' Resources:
1)  "Lognormal Model for Stock Prices", Michael J. Sharpe, UCD Math Dept, 
    http://www.math.ucsd.edu/~msharpe/stockgrowth.pdf
TODO:
    1) Add processing of multiple stocks if num stocks >1
    2) Ensure default of a single direct measurement if num stocks =1
'''
class StockMeasurement():
    def __init__(self,
                 noiseSigma=0,
                 logmode=0, 
                 infiles=['GE-2000-Aug-1_to_2015-September-04.csv']
                 ):
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
#        self.min_rec_num = min(self.num_records)

    def nextMeas(self):
        next_tstamp = next(self.myiter)
        if self.logmode==1:
            # hard coded for ref=0
            ref_meas = np.log(self.dfs[0].loc[next_tstamp]['open'])
        else:
            ref_meas = self.dfs[0].loc[next_tstamp]['open']
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
        return meas
    
    def reset(self):
        self.index = -1

    def setNoise(self, sigma=0):
        self.noiseSigma = sigma
