import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_models import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


class Logit():
    
    def __init__(self, X, y, test_size = 0.3):
        
        self.X = X
        self.y = y
        self.test_size = test_size
        
        self.logit = LogisticRegression()
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size = self.test_size)
        
    
     def fit(self):
        
        params = self.optimal_params.copy()
        
        if params is None:
            
            print('Run grid_search() first, with the corresponding parameters')
        
        lr = LogisticRegression(learning_rate = params.learning_rate[0],
                               max_depth = params.max_depth[0],
                               n_estimators = params.n_estimators[0],
                               colsample_bytree = params.colsample_bytree[0])

        xgb.fit(self.X_train, self.y_train)
        
        self.prediction = xgb.predict(self.X_test)
        self.score = xgb.score(self.X_test, self.y_test)
        
        
        
    def grid_search(self, max_depth_range, learning_rate_range, n_estimators_range, colsample_bytree_range):
        
        
        params = { 'max_depth': max_depth_range,
                   'learning_rate': learning_rate_range,
                   'n_estimators': n_estimators_range,
                   'colsample_bytree': colsample_bytree_range}

        xgbr = XGBRegressor(seed = 20)
        
        gs = GridSearchCV(estimator=xgbr, 
                           param_grid=params,
                           scoring='neg_mean_squared_error', 
                           verbose=1)
        
        
        gs.fit(self.X_train, self.y_train)
        
        self.optimal_params = pd.DataFrame(gs.best_params_, index = [0])