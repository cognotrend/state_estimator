import numpy as np
import random
import csv
import datetime as dt
class StockMeasurement():
    def __init__(self,noiseSigma=0,infile='GE-2000-Aug-1_to_2015-September-04.csv'):
        i=0
        self.noiseSigma = noiseSigma
        self.oprice = []
        self.cprice = []
        self.dates=[]
        colmap = {'Open':1,'High':2,'Low':3,'Close':4,'Volume':5}
        colindx = 1
        with open(infile, newline='') as csvfile:
        #with open('test.csv', newline='') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',')
            for row in csvreader:
                    plottitle='Open'
                    self.dates.append(row[0])
                    self.oprice.append(row[1])
                    self.cprice.append(row[4])
                    i=i+1

        print('Finished reading.')
        csvfile.close

        self.dates.reverse()
        base_date = self.dates[0].split('/')
        base_mon  = int(base_date[0])
        base_day  = int(base_date[1])
        base_year = int(base_date[2])+2000
        start_date = dt.date(base_year,base_mon,base_day)
        sdord = start_date.toordinal()
        date_ordinals = []

        self.oprice.reverse()
        self.cprice.reverse()
        self.index = -1
        self.meas_func = self.nextMeas
        self.meas_array = np.zeros((1,1))

    def nextMeas(self):
        self.index = self.index+1
        self.meas_array[(0,0)]=self.oprice[self.index]
        return self.meas_array + random.gauss(0,self.noiseSigma)

    def reset(self):
        self.index = -1

    def setNoise(self, sigma=0):
        self.noiseSigma = sigma





        
    
