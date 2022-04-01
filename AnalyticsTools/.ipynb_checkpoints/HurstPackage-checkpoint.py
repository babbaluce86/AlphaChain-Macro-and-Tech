import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import warnings
from sklearn.linear_model import LinearRegression
warnings.filterwarnings("ignore")
warnings.simplefilter('ignore', np.RankWarning)



class HurstExponent():
    
    def __init__(self, series, subdivision_limit):
        
        if not isinstance(series, pd.Series):
            series = pd.Series(series)
            
            
        self.linreg = LinearRegression() 
        self.series = series
            
        if not isinstance(subdivision_limit, int):
            raise ValueError(f'subdivision_limit must be a integer, found {subdivision_limit}')
            
        self.subdivision_limit = subdivision_limit
        
        
        
    def fit(self):
        
        N = self.series.shape[0]

        powers = [pow(2,k) for k in range(1, self.subdivision_limit)]

        logn = []
        logRS = []

        for power in powers:

            n, mean_RS = self.data_preprocessing(power) 
            logn.append(np.log(n))
            logRS.append(np.log(mean_RS))
            
        array_RS = np.array(logRS)
        array_RS = array_RS[~np.isnan(array_RS)]
        
        logRS = array_RS.tolist()
        
        if len(logn) != len(logRS):
            
            discrepancy = abs(len(logn) - len(logRS))
            
            logn = logn[discrepancy:]
            
        hurst_extimation = np.polyfit(logn, logRS, 1)
            
        
        #lgn = pd.Series(logn).fillna(method = 'ffill').values
        #lgRS = pd.Series(logRS).fillna(method = 'bfill').values
        
        #fitted = self.linreg.fit(lgn.reshape(-1,1), lgRS)
        
        #return abs(fitted.coef_)
        
        #hurst_extimation = np.polyfit(lgn, lgRS, 1)

        return abs(hurst_extimation[0])    
        
        
    def subsetting(self, k):
        
        def chunks(l, n):
      
            for i in range(0, len(l), n): 
                yield l[i:i + n]

        return list(chunks(self.series, k))
    
    def data_preprocessing(self, k):
    
        if k == 0:

            subset = self.series
            demeaned = subset - subset.mean()
            cumsum = demeaned.cumsum()
            R = max(cumsum) - min(cumsum)
            RS = R / self.series.std()

        else:

        
            subset = self.subsetting(k)

            lengths = [len(x) for x in subset]

            for x in subset:
                if (len(x) < max(lengths)):
                    subset.pop()
        
        demeaned = np.vstack(subset) - np.mean(np.vstack(subset), axis=1).reshape(-1,1)
        std = np.array([np.std(x) for x in subset])
        
        cumsum = np.cumsum(demeaned, axis = 1)
        
        R = np.max(cumsum, axis = 1) - np.min(cumsum, axis = 1)
        
        RS = R / std
        
        return len(subset), np.mean(RS)    
    
    
    
class HurstIndicator(HurstExponent):
    
    def __init__(self, series, subdivision_limit = 4, smooth_filter = None):
        
        self.smooth_filter = smooth_filter
        
        super().__init__(series, subdivision_limit)
        

    
    def fit(self, lags, window = None, plot = False):
        
        data = self.series.copy()
        
        hurst = []
        
        for k in range(len(data)):
            
            try:
                
                hurst.append(HurstExponent(data[k : k + lags], 
                                           subdivision_limit = self.subdivision_limit).fit())
                
            except:
                
                ValueError
                
        discrepancy = data.shape[0] - len(hurst)
        
        indicator = pd.Series(hurst)
        indicator.index = data.index[discrepancy:]
        
        if self.smooth_filter == 'simple':
            
            simple_smoothing = indicator.rolling(window = window).mean()
        
            if plot:
                
                simple_smoothing.plot(figsize = (16,9), grid = True)
            
            return simple_smoothing
                
        elif self.smooth_filter == 'weighted':
            
            weighted_smoothing = indicator.ewm(span = window, adjust = True).mean()
            
            if plot:
                
                weighted_smoothing.plot(figsize = (16,9), grid = True)
            
            return weighted_smoothing
        
        
        if plot:
            
            indicator.plot(figsize = (16,9), grid = True)
            
        return indicator