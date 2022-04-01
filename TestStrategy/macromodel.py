import pandas as pd 
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from TestStrategy.coremodule.vectorbase import VectorBase

import warnings 

warnings.filterwarnings('ignore')


class Forest(VectorBase):
    
    
    def __init__(self, data, training_size):
        
        self.data = data
        self.training_size = training_size
        
        self.data.dropna(inplace = True)
        
        self.X = self.data.drop(columns = ['Close', 'returns'])
        
        self.y = pd.Series(np.where(self.data.returns > 0, 1, 0), index = self.data.returns.index)
        
        self.train_size = math.ceil(self.training_size * self.X.shape[0])
        
        self.X_train, self.X_test = self.X[:self.train_size], self.X[self.train_size:]
        self.y_train, self.y_test = self.y[:self.train_size], self.y[self.train_size:]
        
        
        super().__init__(data, commissions = None)
    
    
    def test_strategy(self, params):
        
        self.on_data(params)
        self.run_backtest()
        
        #Store and show relevant data
        data = self.results.copy()
        self.results = data
        
        self.print_performance()
        
        
    def on_data(self, params):
        
        data = self.data.copy()
        
        X_train, X_test, y_train, y_test = self.X_train, self.X_test, self.y_train, self.y_test
        
        rf = RandomForestClassifier(random_state = params.random_state[0],
                                    n_estimators = params.n_estimators[0],
                                    min_samples_split= params.min_samples_split[0],
                                    min_samples_leaf = params.min_samples_leaf[0],
                                    max_depth = params.max_depth[0])
        
        
        rf.fit(X_train, y_train)
        
        pred = pd.DataFrame(rf.predict_proba(X_test), columns = ['lower', 'upper'], index = y_test.index)
        pred['delta'] = pred.upper - pred.lower
        
        data = data[pred.index[0]:pred.index[-1]]
        
        data['returns'] = np.log(data.Close / data.Close.shift(1))
        data['delta_prob'] = pred.delta
        
        data['position'] = np.sign(data.delta_prob)
        
        data.dropna(inplace = True)
        
        self.score = rf.score(X_test, y_test)
        
        self.results = data
        
        
    def optimize_strategy(self, param_grid):
        
        X_train = self.X_train
        y_train = self.y_train
        
        model = RandomForestClassifier()

        best_rf = RandomizedSearchCV(model, param_grid, cv = 3, verbose = 1, n_jobs = -1)

        best_rf.fit(X_train, y_train)
        
        self.best_params = pd.DataFrame(best_rf.best_params_, index = [0])
        
        self.find_best_strategy()
        
        
    
    def find_best_strategy(self):
        
        best_params = self.best_params
        
        self.test_strategy(params = best_params)


        
        

class Logit(VectorBase):
    
    
    def __init__(self, data, training_size):
        
        self.data = data
        self.training_size = training_size
        
        self.data.dropna(inplace = True)
        
        self.X = self.data.drop(columns = ['Close', 'returns'])
        
        self.y = pd.Series(np.where(self.data.returns > 0, 1, 0), index = self.data.returns.index)
        
        self.train_size = math.ceil(self.training_size * self.X.shape[0])
        
        self.X_train, self.X_test = self.X[:self.train_size], self.X[self.train_size:]
        self.y_train, self.y_test = self.y[:self.train_size], self.y[self.train_size:]
        
        
        super().__init__(data, commissions = None)
    
    
    def test_strategy(self, params):
        
        self.on_data(params)
        self.run_backtest()
        
        #Store and show relevant data
        data = self.results.copy()
        self.results = data
        
        self.print_performance()
        
        
    def on_data(self, params):
        
        data = self.data.copy()
        
        X_train, X_test, y_train, y_test = self.X_train, self.X_test, self.y_train, self.y_test
        
        logit = LogisticRegression(solver = params.solver[0], 
                                   penalty = params.penalty[0], 
                                   max_iter = params.max_iter[0], 
                                   C =  params.C[0])
                                   
        logit.fit(X_train, y_train)
        
        proba = pd.DataFrame(logit.predict_proba(X_test), columns = ['lower', 'upper'], index = y_test.index)
        pred = pd.Series(logit.predict(X_test), index = y_test.index)
        
        proba['delta'] = proba.upper.sub(proba.lower)
        
        data = data[pred.index[0]:pred.index[-1]]
        
        data['returns'] = np.log(data.Close / data.Close.shift(1))
        data['prediction'] = np.where(pred == 1, 1, -1)
        data['delta_prob'] = abs(proba.delta)
        data.loc[data.delta_prob >= 0.03, 'position'] = data.prediction
        
        
        #cond_long = (data.direction > 0) and (data.delta_prob )
        #data['position'] = np.sign(data.delta_prob)
        
        data.dropna(inplace = True)
        
        self.score = logit.score(X_test, y_test)
        self.results = data
        
        
    def optimize_strategy(self, param_grid):
        
        X_train = self.X_train
        y_train = self.y_train
        
        model = LogisticRegression()

        best_rf = RandomizedSearchCV(model, param_grid, cv = 3, verbose = 1, n_jobs = -1)

        best_rf.fit(X_train, y_train)
        
        self.best_params = pd.DataFrame(best_rf.best_params_, index = [0])
        
        self.find_best_strategy()
        
        
    
    def find_best_strategy(self):
        
        best_params = self.best_params
        
        self.test_strategy(params = best_params)
        
        
        
        
        
        
class Linearized(VectorBase):
    
    
    
    def __init__(self, data, training_size):
        
        self.data = data
        self.training_size = training_size
        
        self.data.dropna(inplace = True)
        
        self.X = self.data.drop(columns = ['Close', 'returns'])
        
        
        self.y = self.data.returns
        
        self.train_size = math.ceil(self.training_size * self.X.shape[0])
        
        self.X_train, self.X_test = self.X[:self.train_size], self.X[self.train_size:]
        self.y_train, self.y_test = self.y[:self.train_size], self.y[self.train_size:]
        
        
        super().__init__(data, commissions = None)
    
    
    def test_strategy(self):
        
        self.on_data()
        self.run_backtest()
        
        #Store and show relevant data
        data = self.results.copy()
        self.results = data
        
        self.print_performance()
        
        
    def on_data(self):
        
        data = self.data.copy()
        
        X_train, X_test, y_train, y_test = self.X_train, self.X_test, self.y_train, self.y_test
        
        lr = LinearRegression()
        
        lr.fit(X_train, y_train)
        
        pred = pd.Series(lr.predict(X_test) , index = y_test.index)
        
        data = data[pred.index[0]:pred.index[-1]]
        
        data['returns'] = data.returns
        data['predicted'] = pred
        data['position'] = np.sign(data.predicted)
        
        data.dropna(inplace = True)
        
        self.score = lr.score(X_test, y_test)
        self.results = data
    
    
    


        
        
        
        
        