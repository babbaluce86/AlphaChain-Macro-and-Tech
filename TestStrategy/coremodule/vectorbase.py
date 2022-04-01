from abc import ABCMeta, abstractmethod
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings


plt.style.use('seaborn')

warnings.filterwarnings('ignore')


class VectorBase(object):
    
    '''Base Class for Vectorized Backtesting. Supports any child class
       for simple backtesting. 
    
    
    Parameters: 
    =====================
    
    data: pd.DataFrame
    
    ---------------------
    
    commission: float
    
    ======================
    
    Methods:
    ======================
    
    @abstractmethod
    test_strategy() 
    
    ----------------------
    
    @abstractmethod
    on_data()
    
    ----------------------
    
    @Classmethod
    run_backtest()
    
    ----------------------
    
    @Classmethod
    plot_results()
    
    ----------------------
    
    @Classmethod
    plot_diagnostics()
    
    ----------------------
    
    @abstractmethod
    optimize_strategy()
    
    ----------------------
    
    @abstractmethod
    find_best_strategy()
    
    ----------------------
    
    @Classmethod
    print_performance()
    
    ----------------------
    
    @Classmethod
    calculate_multiple()
    
    ====================== 
    
    '''
    
    def __init__(self, data, commissions = None):
        
        
        self.data = data
        self.commissions = commissions
        
        self.total_days = 365.25
            
        self.results = None    
            
        self.tp_year = (self.data.Close.count() / ((self.data.index[-1] - self.data.index[0]).days / self.total_days))
        
        
        
        
    @abstractmethod
    def test_strategy(self):
        '''This method should be implemented in a child class based upon the on_data() method below. Tests the strategy
            with the rationale given in the on_data() method.'''
        raise NotImplementedError("Should implement test_strategy()")
    
    @abstractmethod
    def on_data(self):
        '''This method should be implemented in a child class, 
            it provides the data manuevering of the strategy's rationale'''
        raise NotImplementedError("Should implement on_data()")
        
    def run_backtest(self):
        ''' Runs the strategy backtest.
        '''    
        data = self.results.copy()
        data["sreturns"] = data["position"].shift(1) * data["returns"]
        data["trades"] = data.position.diff().fillna(0.0).abs()
        
        
        if self.commissions:
            
            data.strategy = data.strategy + data.trades * self.commissions
        
        data['csreturns'] = data['sreturns'].cumsum().apply(np.exp)
        data['drawdown'] = data['csreturns'] - data['csreturns'].cummax()
        
        
        self.results = data
        
        
        
        
    def plot_results(self):
        
        '''Plots the results of the test_strategy method against the naive benchmark buy and hold'''
        
        if self.results is None:
            print('Run test_strategy first')
        
        else:
            
            title = 'Strategy Performance'
            
            self.results['csreturns'].plot(figsize = (16,9), title = title)
            
            
    def plot_diagnostics(self, no_log = False):
        
        if self.results is None:
            print('Run test_strategy first')
        
        else:
            
            res = self.results.copy()
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
            fig.suptitle('Distributions of Strategy Returns')

            # Returns
            sns.histplot(ax = axes[0], data = np.log(np.ma.masked_invalid(res.sreturns)), kde = True)
            axes[0].set_title('Strategy Returns')

            # Long Positions
            if no_log:
                
                # Long Positions
                sns.histplot(ax = axes[1], data = np.ma.masked_invalid(res.query('trades !=0 & position == 1.0').sreturns), kde = True)
                axes[1].set_title('Long Positions')
                
                # Short Positions
                sns.histplot(ax = axes[2], data = np.ma.masked_invalid(res.query('trades !=0 & position == -1.0').sreturns), kde = True)
                axes[2].set_title('Short Positions')
                
            else:
                
                #Long Positions
                sns.histplot(ax = axes[1], data = np.log(np.ma.masked_invalid(res.query('trades !=0 & position == 1.0').sreturns)), kde = True)
                axes[1].set_title('Long Positions')

                # Short Positions
                sns.histplot(ax = axes[2], data = np.log(np.ma.masked_invalid(res.query('trades !=0 & position == -1.0').sreturns)), kde = True)
                axes[2].set_title('Short Positions')
            
            
            plt.show()
    
        
    @abstractmethod
    def optimize_strategy(self):
        '''This method should be implemented when the given strategy parameters are tuned to maximize 
           returns or sharpe ratio of the strategy '''
        raise NotImplementedError("Should implement optimize_strategy()")
        
    @abstractmethod
    def find_best_strategy(self):
        '''This method should be implemented as a optimal parameter grid search based on the 
            optimize_strategy() method'''
        raise NotImplementedError("Should implement best_strategy()")
        
   
             
        
    def plot_diagnostics(self, no_log = False):
        
        if self.results is None:
            print('Run test_strategy first')
        
        else:
            
            res = self.results.copy()
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
            fig.suptitle('Distributions of Strategy Returns')

            # Returns
            sns.histplot(ax = axes[0], data = np.log(np.ma.masked_invalid(res.sreturns)), kde = True)
            axes[0].set_title('Strategy Returns')

            # Long Positions
            if no_log:
                
                # Long Positions
                sns.histplot(ax = axes[1], data = np.ma.masked_invalid(res.query('trades !=0 & position == 1.0').sreturns), kde = True)
                axes[1].set_title('Long Positions')
                
                # Short Positions
                sns.histplot(ax = axes[2], data = np.ma.masked_invalid(res.query('trades !=0 & position == -1.0').sreturns), kde = True)
                axes[2].set_title('Short Positions')
                
            else:
                
                #Long Positions
                sns.histplot(ax = axes[1], data = np.log(np.ma.masked_invalid(res.query('trades !=0 & position == 1.0').sreturns)), kde = True)
                axes[1].set_title('Long Positions')

                # Short Positions
                sns.histplot(ax = axes[2], data = np.log(np.ma.masked_invalid(res.query('trades !=0 & position == -1.0').sreturns)), kde = True)
                axes[2].set_title('Short Positions')
            
            
            plt.show()
        
        
        
    def print_performance(self):
        ''' Calculates and prints various Performance Metrics.
        '''
        
        data = self.results.copy()
        
        
        n_trades = data.trades.sum()
        winners = data.query('trades != 0 & sreturns > 0').shape[0] 
        loosers = n_trades - winners
        win_ratio = round(winners / n_trades, 1)
        loose_ratio = round(1.0 - win_ratio, 1)
        maximum_drawdown = round(data.drawdown.min(), 4)
        terminal_wealth = round(data.csreturns[-1], 4)
        sharpe = round(data.sreturns.mean() / data.sreturns.std(), 4)
        
        
        print(100 * "=")
        print("STRATEGY PERFORMANCE")
        print(100 * "-")
        print('TRADING PERFORMANCE MEASURES:')
        print('\n')
        print('Number of Trades: {}'.format(n_trades))
        print('Number of Winners: {}'.format(winners))
        print('Number of Loosers: {}'.format(loosers))
        print('\n')
        print('Win Ratio: {}'.format(win_ratio))
        print('Loose Ratio: {}'.format(loose_ratio))
        print("Sharpe Ratio:                {}".format(sharpe))
        
        print('\n')
        print('Terminal Wealth: {}'.format(terminal_wealth))
        print('Maximum Drawdown: {}'.format(maximum_drawdown))
        
        print(100 * "=")
        
    def calculate_multiple(self, series):
        '''This is simply implied when calculating the multiple of the strategy returns, i.e.
            as if the trader would invest the amount of 1 dollar. '''
        return np.exp(series.sum())
    