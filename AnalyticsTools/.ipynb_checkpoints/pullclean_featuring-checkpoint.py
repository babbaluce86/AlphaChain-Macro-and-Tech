import json
import requests
import pandas as pd
import numpy as np
import math
from datetime import datetime

API_KEY = '26eAaUMUEzN6jbOpvRDjj0C9Q9V'

class GetnClean():
    
    def __init__(self, directory, data_name, start_date):
        
        #self.symbol = symbol
        self.directory = directory
        self.data_name = data_name 
        self.start_date = start_date
        
        
        self.get_data()
        self.clean_data()
        
        
    def get_data(self):
        
        ts = pd.to_datetime(self.start_date).timestamp()
        
        response = requests.get(f'https://api.glassnode.com/v1/metrics/{self.directory}/{self.data_name}', 
                                params = {'a': 'BTC', 
                                          's': int(ts),
                                          'i': '24h',
                                          'api_key': API_KEY})
        
        self.raw_data = pd.DataFrame(json.loads(response.text))

        
    def clean_data(self):
        
        data = self.raw_data.copy()
        
        date = pd.to_datetime(data.t, unit = 's')
        values = pd.to_numeric(data.v)
        
        series = pd.Series(data = values.values, index = date)
        
        self.data = series.to_frame(f'{self.data_name}')
        
        
        
        
        
# convert series to supervised learning
def create_lagsnleads(data, n_lag=1, n_lead=1, dropnan=True):
    
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    
    # Create Lags
    
    for i in range(n_lag, 0, -1):
        
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
     
    
    # Create Leads
    
    for i in range(0, n_lead):
        
            
        cols.append(df.shift(-i))
        if i == 0:
            
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        
        else:
            
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
     # put it all together
     
    agg = pd.concat(cols, axis=1)
    agg.columns = names
     # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    
    return agg