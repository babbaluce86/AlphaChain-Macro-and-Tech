import pandas as pd
import numpy as np
import math
import calendar

import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import MinMaxScaler, StandardScaler

import tensorflow as tf
from tensorflow import keras

np.random.seed(42)
tf.random.set_seed(42)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, RepeatVector, TimeDistributed

plt.style.use('seaborn')


class FinancialAnalysis(): #Parent Class
    
    def __init__(self, ticker, start, end, interval = None, filepath = None):
        
        self._ticker = ticker
        self.start = start
        self.end = end
        self.interval = interval
        self.filepath = filepath
        self.get_data()
        self.log_returns()
        
    def __repr__(self):
        return 'FinancialAnalysis(symbol = {}, start = {}, end = {}, interval = {}, filepath = {}'.format(self._ticker, self.start, self.end, self.interval, self.filepath)
        
    def get_data(self):
        
        if self.filepath:
            
            raw = pd.read_csv(self.filepath, parse_dates = ["Date"], index_col = "Date")
            self.data = raw[self.start : self.end].Close
        
        elif self.interval:
            
            columns = ['Close','Volume']
            self.data = yf.Ticker(self._ticker).history(start = self.start, 
                                                        end = self.end, 
                                                        interval = str(self.interval))[columns]
            
    
    def plot_prices(self):
        
        plt.figure(figsize=(16,8))
        
        plt.plot(self.data.Close)
        plt.axhline(self.data.Close.mean(), color = 'r')
        
        plt.ylabel('Price')
        plt.xlabel('Date')
        
        plt.tick_params(axis = 'both', which = 'major', labelsize=15)
        plt.title('Price Chart : {}'.format(self._ticker))
        plt.suptitle("")
        
        plt.show()
    
    def log_returns(self):
        self.data['log_returns'] = np.log(self.data.Close/self.data.Close.shift(1))
        
    def plot_returns(self, hist):
        
        plt.figure(figsize=(16,8))
        
        if hist:
            
            plt.hist(self.data.log_returns, bins=100)
            
            plt.ylabel('Frequency Returns')
            plt.xlabel('Date')
            plt.tick_params(axis = 'both', which = 'major', labelsize=15)
            plt.title('Frequency of Returns : {}'.format(self._ticker))
            plt.suptitle("")

            plt.show()

        else:
            
            plt.plot(self.data.log_returns)
            
            plt.ylabel('log Returns')
            plt.xlabel('Date')
            plt.tick_params(axis = 'both', which = 'major', labelsize=15)
            plt.title('logReturns Chart : {}'.format(self._ticker))
            plt.suptitle("")

            plt.show()

    
    def set_ticker(self, ticker = None):
        if ticker is not None:
            self._ticker = ticker
            self.get_data()
            self.log_returns()
            
######################################################################################           
            
class RiskReturn(FinancialAnalysis): #Child Class
    
    def __init__(self, ticker, start, end, interval, freq = None):
        self.freq = freq
        super().__init__(ticker, start, end, interval) #overwrites from the Parent Class
    
    def __repr__(self):
        return 'RiskReturn(symbol = {}, start = {}, end = {}, interval = {}, filepath = {}'.format(self._ticker, self.start, self.end, self.interval, self.filepath)
    
    
    def mean_returns(self):
        
        if self.freq is None:
            return self.data.log_returns.mean()
        
        else:
            resampled_price = self.data.Close.resample(self.freq).last()
            resampled_returns = np.log(resampled_price/resampled_price.shift(1))
            return resampled_returns.mean()
        
    
    def std_returns(self):
        
        if self.freq is None:
            return self.data.log_returns.std()
        
        else:
            resampled_price = self.data.Close.resample(self.freq).last()
            resampled_returns = np.log(resampled_price/resampled_price.shift(1))
            return resampled_returns.std()
        
    def annualized_return(self, instrumentType='stock'):
        
        if (instrumentType == 'crypto') or (instrumentType == 'energyCommodity'):
            mean_return = round(self.data.log_returns.mean()*365.25,3)
            return mean_return
        
        else:
            mean_return = round(self.data.log_returns.mean()*252,3)
            return mean_return
        
    def annualized_std(self, instrumentType = 'stock'):
        
        if (instrumentType == 'crypto') or (instrumentType == 'energyCommodity'):
            risk = round(self.data.log_returns.std()*np.sqrt(252),3)
            return risk
        
        else:
            risk = round(self.data.log_returns.std()*np.sqrt(252),3)
            return risk
        
        
    def sharpe_ratio(self, instrumentType = 'stock'):
        return self.annualized_return()/self.annualized_std()
    
    def drawdown(self):
        
        multiple = self.data.log_returns.cumsum().apply(np.exp).dropna()
        previous_peaks = multiple.cummax()
        drawdowns = (multiple - previous_peaks)/previous_peaks
        return pd.DataFrame({'Wealth': multiple,
                             'Previous Peak': previous_peaks,
                             'Drawdown': drawdowns})
    
    def var_historic(self, level=5):
        '''Computes the historic Value at Risk at a specified level, i.e.
           returns the number such that the "level" percent of the returns fall
           below that number, and the (100-level) percent are above'''
        return -np.percentile(self.data.log_returns.dropna(), level)
    
    def cvar_historic(self, level=5):
        '''Computes the Conditional VaR of the returns'''
        is_beyond = self.data.log_returns.dropna() <= -self.var_historic()
        return self.data.log_returns.dropna()[is_beyond == True].mean()
    
    def performance_summary(self, instrumentType = 'stock'):
        
        ann_rets = self.annualized_return(instrumentType=instrumentType)
        ann_vol = self.annualized_std(instrumentType=instrumentType)
        ann_sr = self.sharpe_ratio(instrumentType = instrumentType)
        dd = self.drawdown().Drawdown.min()
        var = self.var_historic()
        cvar = self.cvar_historic()
        
        print(100 * "=")
        print("PERFORMANCE SUMMARY | INSTRUMENT = {} |".format(self._ticker))
        print(100 * "-")
        print("PERFORMANCE MEASURES:")
        print("\n")
        print("Annualized Returns:         {}".format(ann_rets))
        print("Annualized Volatility:     {}".format(ann_vol))
        print("Annualized Sharpe Ratio:    {}".format(ann_sr))
        print("Maximum Drawdown:           {}".format(dd))
        print(38 * "-")
        print("\n")
        print("Value at Risk:       {}".format(var))
        print("Conditional Value ar Risk      {}".format(cvar))
        print(100 * "=")
        
        
#######################################################################################

class Statistics(FinancialAnalysis):
    
    
    def __repr__(self):
        return 'Statistics(symbol = {}, start = {}, end = {}, interval = {}, filepath = {}'.format(self._ticker, self.start, self.end, self.interval, self.filepath)
    
    
    def weekly_averages(self, plot = True):
        
        data_price = self.data.Close.copy()
        data_volume = self.data.Volume.copy()/1000000000
        data_returns = self.data.log_returns.copy()
        
        
        unstackPrice_wk = self.data_unstack(data_price, by = 'week')
        unstackVolume_wk = self.data_unstack(data_volume, by = 'week')
        unstackRets_wk = self.data_unstack(data_returns, by = 'week')
        
        self.week_price = self.summary_stats(unstackPrice_wk)
        self.week_volume = self.summary_stats(unstackVolume_wk)
        self.week_returns = self.summary_stats(unstackRets_wk)
        
        if plot:
            
            fig, (ax1, ax2, ax3) = plt.subplots(3, figsize = (20,20))
        
            
            unstackPrice_wk.boxplot(ax=ax1)
            ax1.set_ylabel('Price')
            plt.tick_params(axis = 'both', which = 'major', labelsize=15)
            ax1.set_title('Daily Average Price in a week')
            
            unstackRets_wk.boxplot(ax=ax2)
            ax2.set_ylabel('Returns')
            plt.tick_params(axis = 'both', which = 'major', labelsize=15)
            ax1.set_title('Daily Average Return in a week')
            
            unstackVolume_wk.boxplot(ax=ax3)
            ax3.set_ylabel('Volume')
            plt.tick_params(axis = 'both', which = 'major', labelsize=15)
            ax1.set_title('Daily Average Volume in a week')
            
            
            plt.show()

        
        
    def monthly_averages(self, plot = True):
        
        data_price = self.data.Close.copy()
        data_volume = self.data.Volume.copy()
        data_returns = self.data.log_returns.copy()
        
        unstackPrice_mo = self.data_unstack(data_price, by = 'month')
        unstackVolume_mo = self.data_unstack(data_volume, by = 'month')
        unstackRets_mo = self.data_unstack(data_returns, by = 'month')
        
        self.week_price = self.summary_stats(unstackPrice_mo)
        self.week_volume = self.summary_stats(unstackVolume_mo)
        self.week_returns = self.summary_stats(unstackRets_mo)
        
        if plot:
            
            fig, (ax1, ax2, ax3) = plt.subplots(3, figsize = (20,20))
        
            
            unstackPrice_mo.boxplot(ax=ax1)
            ax1.set_ylabel('Price')
            plt.tick_params(axis = 'both', which = 'major', labelsize=15)
            ax1.set_title('Daily Average Price in a month')
            
            unstackRets_mo.boxplot(ax=ax2)
            ax2.set_ylabel('Returns')
            plt.tick_params(axis = 'both', which = 'major', labelsize=15)
            ax1.set_title('Daily Average Return in a month')
            
            unstackVolume_mo.boxplot(ax=ax3)
            ax3.set_ylabel('Volume')
            plt.tick_params(axis = 'both', which = 'major', labelsize=15)
            ax1.set_title('Daily Average Volume in a month')
            
            
            plt.show()

  
        
    def data_unstack(self, df, by):
        
        if by == 'week':
            
            unstacked = df.groupby([df.index.hour, df.index.weekday]).mean().unstack(level = -1)
            unstacked.columns = list(calendar.day_name)
            
        elif by == 'month':
            
            unstacked = df.groupby([df.index.hour, df.index.day]).mean().unstack(level = -1)
            
        return unstacked
        
        
        
    def summary_stats(self, df):
        
        description = df.describe()
        description.loc['when_min'] = df.idxmin()
        description.loc['when_max'] = df.idxmax()
        
        return description
    



class DetectAnomaly(FinancialAnalysis):
    
        
    def __init__(self, ticker, start, end, interval, time_steps):
        
        self.time_steps = time_steps
        
        super().__init__(ticker, start, end, interval) #overwrites from the Parent Class
    
        self.scaler = StandardScaler()
        self.prepare_data()
        self.train_test_subsets()
        self.autoencoder()
        self.get_score_df()
        
    def __repr__(self):
        return 'DetectAnomaly(ticker = {}, start = {}, end = {}, interval = {}, filepath = {}'.format(self._ticker, self.start, self.end, self.interval, self.filepath)
        
    
    def prepare_data(self):
        
        data = self.data.copy()
        
        size = math.ceil(len(data)*.95)

        self.train, self.test = data.Close.iloc[0:size].to_frame('close'), data.Close.iloc[size:len(data)].to_frame('close')
        
        
        self.scaler.fit(self.train)

        self.train['close'] = self.scaler.transform(self.train[['close']])
        self.test['close'] = self.scaler.transform(self.test[['close']])
    
    def create_dataset(self, X, y):
    
        Xs, ys = [], []

        for k in range(len(X)-self.time_steps):
            Xs.append(X.iloc[k:(k+self.time_steps)].values)
            ys.append(y.iloc[k+self.time_steps])

        return np.array(Xs), np.array(ys)
    
    def train_test_subsets(self):
        
        self.X_train, self.y_train = self.create_dataset(self.train[['close']], self.train['close'])
        self.X_test, self.y_test = self.create_dataset(self.test[['close']], self.test['close'])
        
    def autoencoder(self):
        
        X_train = self.X_train.copy()
        X_test = self.X_test.copy()
        y_train = self.y_train.copy()
        y_test = self.y_test.copy()
        
        model = Sequential()
        model.add(LSTM(units = 64, input_shape = (X_train.shape[1], X_train.shape[2])))
        model.add(Dropout(rate=0.2))
        model.add(RepeatVector(n=X_train.shape[1]))
        model.add(LSTM(units = 64, return_sequences=True, ))
        model.add(Dropout(rate=0.2))
        model.add(TimeDistributed(Dense(units = X_train.shape[2])))

        model.compile(loss = 'mae', optimizer = 'adam')
        
        history = model.fit(X_train, y_train, epochs=10, batch_size=32, 
                    validation_split=0.1,
                    callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, mode='min')], 
                    shuffle=False)
        
        
        self.X_test_pred = model.predict(X_test)

        self.test_mae_loss = np.mean(np.abs(self.X_test_pred, X_test), axis=1)
        
        
    def get_threshold(self):
        
        test_mae_loss = self.test_mae_loss.copy()
        scores = test_mae_loss.copy()
        scores.sort()
        cut_off = int(.8*len(scores))
        threshold = scores[cut_off]
        
        return float(threshold)
    
    def get_score_df(self):
        
        
        self.test_score_df = pd.DataFrame(index=self.test[self.time_steps:].index)
        self.test_score_df['loss'] = self.test_mae_loss.copy()
        self.test_score_df['threshold'] = self.get_threshold()
        self.test_score_df['anomaly'] = self.test_score_df.loss > self.test_score_df.threshold
        self.test_score_df['close'] = self.test[self.time_steps:].close
        
        self.anomalies = self.test_score_df[self.test_score_df.anomaly == True]
        
    def plot_anomalies(self):
        
        
        plt.figure(figsize = (16,9))
        
        
        plt.plot(
            
            self.test[self.time_steps:].index, 
            self.scaler.inverse_transform(self.test[self.time_steps:].close), 
            label='close price'
            
        );

        sns.scatterplot(
              self.anomalies.index,
              self.scaler.inverse_transform(self.anomalies.close),
              color=sns.color_palette()[1],
              s=52,
              label='anomaly'
            )
        plt.xticks(rotation=25)
        plt.legend()
        plt.show();    
    

        
        
    
        
        
        