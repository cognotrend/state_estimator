#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 14:54:50 2020

@author: mcarroll
"""

import datetime as dt
import pandas as pd
import numpy as np

infiles=['daily_adjusted_IBM.csv',
         'daily_adjusted_AMZN.csv',
         'daily_adjusted_GE.csv']
tickers=['IBM','AMZN','GE']

d_parser = lambda x: dt.datetime.fromisoformat(x)

dfs = []
for stock_file in infiles:
    df = pd.read_csv(stock_file, 
                     index_col='timestamp', 
                     parse_dates=['timestamp'],
                     date_parser=d_parser)
    # Reverse data coming from Alpha Vantage
    df = df[::-1]
    dfs.append(df)
    
timestamps = dfs[0].index

myiter = iter(timestamps)
#ts = next(myiter)

for x in list(range(5)):
    ts = next(myiter)
    for i in [0,1,2]:
        print(tickers[i])
        print(np.log(dfs[i].loc[ts][['open','close']]))
    



