import pandas as pd
import numpy as np
import yfinance as yf
import math
from binance.client import Client
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")


api_key = "MxSmc6ia6BA6BA60invAnXfoSolR8sigdbN9d8pCeSwlPq9VhqbovNrCHnp5rgV9"
secret_key = "6OezC4y724zTYDXKCyxfACXlUa3VKJjzZQG6RM79xdmph5d50fiIBaTE2nn9G4vL"


client = Client(api_key = api_key, api_secret = secret_key, tld = "com", testnet = True) # Testnet!!!




class MarketSentiment():
    
    def __init__(self, symbol, bar_length, 
                       start = None, end = None, 
                       filepath = None,
                       yahoo = False, 
                       binance = False):
        
        
        self.symbol = symbol
        self.bar_length= bar_length
        self.start = start
        self.end = end
        self.filepath = filepath
        self.yahoo = yahoo
        self.binance = binance
        
        self.get_data()
        self.overall_trend()
        
    def get_data(self):
        
        
        if self.filepath:
            
            raw = pd.read_csv(self.filepath, parse_dates = ['Date'], index_col = 'Date')[['Open', 'High', 'Low', 'Close', 'Volume']]
            raw.index = pd.to_datetime(raw.index, utc = True)
            
            self.bars = raw
        
        
        elif self.yahoo:

            small_bars = ['5m', '15m', '30m']
            ticker = yf.Ticker(str(self.symbol))

            if self.bar_length == '1h':

                self.bars = ticker.history('3mo', '1h')[['Open', 'High', 'Low', 'Close', 'Volume']]


            elif self.bar_length in small_bars:

                self.bars = ticker.history('60d', self.bar_length)[['Open', 'High', 'Low', 'Close', 'Volume']] 

            elif self.start:

                self.bars = ticker.history(self.start, '1d')[['Open', 'High', 'Low', 'Close', 'Volume']]

            self.bars['log_change'] = np.log(self.bars.Close / self.bars.Close.shift(1))

        
        elif self.binance:
            
            
            bars = client.get_historical_klines(symbol = self.symbol, interval = self.bar_length,
                                        start_str = self.start, end_str = self.end, limit = 1000)
            df = pd.DataFrame(bars)
            df["Date"] = pd.to_datetime(df.iloc[:,0], unit = "ms")
            df.columns = ["Open Time", "Open", "High", "Low", "Close", "Volume",
                          "Clos Time", "Quote Asset Volume", "Number of Trades",
                          "Taker Buy Base Asset Volume", "Taker Buy Quote Asset Volume", "Ignore", "Date"]
            df = df[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()
            df.set_index("Date", inplace = True)
            for column in df.columns:
                df[column] = pd.to_numeric(df[column], errors = "coerce")
                
            self.bars = df
            self.bars['log_change'] = np.log(self.bars.Close / self.bars.Close.shift(1))

        
            
    def overall_trend(self):
        
        data = self.bars.copy()
        
        cond_up = data.Close > data.Open
        cond_down = data.Open > data.Close
        
        amplitude = data.Close - data.Open

        data.loc[cond_up, 'trend'] = 1
        data.loc[cond_down, 'trend'] = -1 
        
        data['amplitude'] = amplitude
        
        self.up_total = data.query('trend == 1').trend.sum()
        self.down_total = abs(data.query('trend == -1').trend.sum())
        
        self.avgup_amplitude = data.query('amplitude > 0').amplitude.mean()
        self.avgdown_amplitude = data.query('amplitude < 0').amplitude.mean()
        
        
        
        print('Cryptocurrency:{}'.format(self.symbol))
        print('-'*100)
        print('Total Examinated Candles: {}'.format(data.shape[0]))
        print('='*100)
        print('Trend Up Total Ratio: {}'.format(round(self.up_total / data.shape[0],4)))
        print('\n')
        print('Trend Down Total Ratio: {}'.format(round(self.down_total / data.shape[0],4)))
        print('='*100)
        print('Average upper candles movement: {}'.format(round(self.avgup_amplitude, 4)))
        print('\n')
        print('Average lower candles movement: {}'.format(round(self.avgdown_amplitude, 4)))
        print('-'*100)
        
    
    def three_candles(self, volatility_window = 6):
        
        data = self.bars.copy()
        close = data.Close
        
        for i in range(3, len(data)):
            
            if (close[i] > close[i-1]) and \
               (close[i-1] > close[i-2]) and \
               (close[i-2] > close[i-3]):
                
                data.loc[data.index[i], 'three_up'] = 1
                #data.loc[data.index[i], 'cumulative_up'] = data.log_change[i].sum()
                #data.loc[data.index[i], 'volume_up'] = data.Volume[i].sum()
                
            elif (close[i] < close[i-1]) and \
                 (close[i-1] < close[i-2]) and \
                 (close[i-2] < close[i-3]):
                
                data.loc[data.index[i], 'three_down'] = 1
                #data.loc[data.index[i], 'cumulative_down'] = data.log_change[i].sum()
                #data.loc[data.index[i], 'volume'] = data.Volume[i].sum()
            
        up_count = data.three_up.sum()
        down_count = data.three_down.sum()
        
        price_up = data.query('three_up == 1').log_change
        price_down = data.query('three_down == 1').log_change
        
        self.price_up = price_up
        
        volume_up = data.query('three_up == 1').Volume
        volume_down = data.query('three_down == 1').Volume
        
        
        data['volatility'] = data.log_change.rolling(window=volatility_window).std()
        
        self.data_three_candles = data
        
        print('Three Candles Analysis | Cryptocurrency: {}'.format(self.symbol))
        print('-'*100)
        print('Consecutive 3 candles up: {}'.format(up_count))
        print('-'*100)
        print('Consecutive 3 candles down: {}'.format(down_count))

        print('='*100)

        print('Overall up ratio: {}'.format(round(up_count / data.shape[0], 4)))
        print('-'*100)
        print('Overall down ratio: {}'.format(round(down_count / data.shape[0], 4)))

        print('='*100)

        print('Up concentration: {}'.format(round(up_count / self.up_total, 4)))
        print('-'*100)
        print('Down concentration: {}'.format(round(down_count / self.down_total, 4)))

        print('='*100)
        
        print('Average price movement up: {} | Impact to average {}'.format(round(price_up.mean(), 4), 
                                                                            round((price_up.mean() / self.avgup_amplitude)**(-1), 4)))
        print('-'*100)
        print('Standard deviation up: {}'.format(round(price_up.std(), 4)))
        print('-'*100)
        print('Average traded volume up: {} | Average volume: {}'.format(volume_up.mean(),
                                                                        self.data_three_candles.Volume.mean()))
        
        print('='*100)
        
        
        print('Average price movement down: {} | Impact to average {}'.format(round(price_down.mean(),4), 
                                                                              abs(round((price_down.mean() / self.avgup_amplitude)**(-1), 4))))
        print('-'*100)
        print('Standard deviation down: {}'.format(round(price_down.std(),4)))
        print('-'*100)
        print('Average traded volume down: {} | Average volume: {}'.format(self.data_three_candles.query('three_down == 1').Volume.mean(),
                                                                           self.data_three_candles.Volume.mean()))
        
        print('-'*100)

    
    def five_candles(self, volatility_window = 6):
        
        data = self.bars.copy()
        data['log_change'] = np.log(data.Close / data.Close.shift(1))
        
        close = data.Close
        
        for i in range(5, len(data)):
            
            if (close[i] > close[i-1]) and \
               (close[i-1] > close[i-2]) and \
               (close[i-2] > close[i-3]) and \
               (close[i-3] > close[i-4]) and \
               (close[i-4] > close[i-5]):
                
                data.loc[data.index[i], 'five_up'] = 1
                data.loc[data.index[i], 'cumulative_up'] = data.log_change[i].sum()
                data.loc[data.index[i], 'volume_up'] = data.Volume[i].sum()
                
                
            elif (close[i] < close[i-1]) and \
                 (close[i-1] < close[i-2]) and \
                 (close[i-2] < close[i-3]) and \
                 (close[i-3] < close[i-4]) and \
                 (close[i-4] < close[i-5]):
                
                data.loc[data.index[i], 'five_down'] = 1
                data.loc[data.index[i], 'cumulative_down'] = data.log_change[i].sum()
                data.loc[data.index[i], 'volume'] = data.Volume[i].sum()
            
            
        up_count = data.five_up.sum()
        down_count = data.five_down.sum()
        
        price_up = data.query('five_up == 1').log_change
        price_down = data.query('five_down == 1').log_change
        
        volume_up = data.query('five_up == 1').Volume
        volume_down = data.query('five_down == 1').Volume
        
        data['volatility'] = data.log_change.rolling(window=volatility_window).std()
        
        self.data_five_candles = data
        
        print('Five Candles Analysis | Cryptocurrency: {}'.format(self.symbol))
        print('-'*100)
        
        print('Consecutive 5 candles up: {}'.format(up_count))
        print('-'*100)
        print('Consecutive 5 candles down: {}'.format(down_count))
        
        print('='*100)
        
        print('Overall up ratio: {}'.format(round(up_count / data.shape[0], 4)))
        print('-'*100)
        print('Overall down ratio: {}'.format(round(down_count / data.shape[0], 4)))
        
        print('='*100)
        
        print('Up concentration: {}'.format(round(up_count / self.up_total, 4)))
        print('-'*100)
        print('Down concentration: {}'.format(round(down_count / self.down_total, 4)))
        
        print('='*100)
                
        print('Average price movement up: {} | Impact to average {}'.format(round(price_up.mean(), 4), 
                                                                            round((price_up.mean() / self.avgup_amplitude)**(-1), 4)))
        print('-'*100)
        print('Standard deviation up: {}'.format(round(price_up.std(), 4)))
        print('-'*100)
        print('Average traded volume up: {} | Average volume: {}'.format(volume_up.mean(),
                                                                        self.data_five_candles.Volume.mean()))
        
        print('='*100)
        
        
        print('Average price movement down: {} | Impact to average {}'.format(round(price_down.mean(),4), 
                                                                              abs(round((price_down.mean() / self.avgup_amplitude)**(-1), 4))))
        print('-'*100)
        print('Standard deviation down: {}'.format(round(price_down.std(), 4)))
        print('-'*100)
        print('Average traded volume down: {} | Average volume: {}'.format(volume_down.mean(),
                                                                        self.data_five_candles.Volume.mean()))
        
        print('-'*100)

    
        
    
