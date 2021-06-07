'''
    File name: init1.py
    Author: Michael Carroll
    Date created: 12/10/2020
    Date last modified: 6/07/2021
    Python Version: 3.7.6
    Description:  Class def for Measurement object.  To handle JSON types.
    Can produce output such as meas.txt:
    
    Current Issues:  
    2021-06-07:  Incomplete.  
    TODO:
    Make clone to work with crypto currency JSON files.
'''

import numpy as np
import datetime as dt
import pandas as pd
import random
import kalman_utils as ku
import json
from json import JSONEncoder

class Measurement():
    def __init__(self, 
                sensor_id='NA', 
                vecshape=(1,1),
                time_of_validity=None, 
                readfile=None, 
                writefile='meas.txt'):
        self.sensor_id = sensor_id
        self.vecshape=vecshape
        self.meas_vec = np.zeros(self.vecshape)
        if time_of_validity==None:
            self.tov = dt.datetime.today()
        self.writefile=writefile
        self.readfile=readfile

    def dumps(self):
        f=open(self.writefile,'a')
        numpy_data = self.meas_vec
        encoded_array = json.dumps(numpy_data, cls=ku.NumpyArrayEncoder)  # use dump() to write array into file
        self.meas_dict = {'sensor_id': self.sensor_id, 'shape': self.vecshape, 'meas': encoded_array, 'tov': str(self.tov)}
        json.dump(self.meas_dict,f)
        f.write('\n')
        f.close()


