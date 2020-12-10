import numpy as np
import datetime as dt
import pandas as pd
import random
import kalman_utils as ku
import json
from json import JSONEncoder

class Measurement():
    def __init__(self, sensor_id='NA', vecshape=(1,1),time_of_validity=None, readfile=None, writefile='meas.txt'):
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
        f.close()


