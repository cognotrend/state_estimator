import numpy as np
import csv
import datetime as dt
''' Resources:
1)  "Lognormal Model for Stock Prices", Michael J. Sharpe, UCD Math Dept, 
    http://www.math.ucsd.edu/~msharpe/stockgrowth.pdf
'''
class StockMeasurement():
    def __init__(self,noiseSigma=0,
                 logmode=0, 
                 infiles=['GE-2000-Aug-1_to_2015-September-04.csv']
                 ):
        self.num_stocks = len(infiles)
        
        i=0
        self.logmode = logmode
        self.noiseSigma = noiseSigma
        self.dates=[]
        self.oprice=[]
        self.cprice=[]
        self.num_records=[]
#        colmap = {'Open':1,'High':2,'Low':3,'Close':4,'Volume':5}
#        colindx = 1
        for stock_file in infiles:
            oprice = []
            cprice = []
            dates=[]
            with open(stock_file, newline='') as csvfile:
            #with open('test.csv', newline='') as csvfile:
                csvreader = csv.reader(csvfile, delimiter=',')
                for row in csvreader:
    #                    plottitle='Open'
                        dates.append(row[0])
                        oprice.append(row[1])
                        cprice.append(row[4])
                        i=i+1
    
            print('Finished reading measurements.')
            csvfile.close
    
            dates.reverse()

            base_date = dates[0].split('/')
            base_mon  = int(base_date[0])
            base_day  = int(base_date[1])
            base_year = int(base_date[2])+2000
            start_date = dt.date(base_year,base_mon,base_day)
    
            oprice.reverse()
            cprice.reverse()
            self.dates.append(dates)
            self.oprice.append(oprice)
            self.cprice.append(cprice)
            self.num_records.append(len(dates))
            
        self.index = -1
        self.meas_func = self.nextMeas
        self.meas_array = np.zeros((self.num_stocks,1))
        self.min_rec_num = min(self.num_records)

    def nextMeas(self):
        self.index = self.index+1
        j=0
        while j<self.num_stocks: 
            self.meas_array[j,0]=self.oprice[j][self.index]
            j=j+1
        noise_array = np.random.normal(0,self.noiseSigma,(self.num_stocks,self.min_rec_num))
        if self.logmode==1:
            return np.log(self.meas_array) + noise_array[:,self.index]
        else:
            return self.meas_array + noise_array[:,self.index]

    def reset(self):
        self.index = -1

    def setNoise(self, sigma=0):
        self.noiseSigma = sigma
