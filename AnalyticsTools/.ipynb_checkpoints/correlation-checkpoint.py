import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from arch import arch_model

import statsmodels.api as sm
from scipy.optimize import fmin, minimize
from scipy.stats import t
from scipy.stats import norm
from math import inf
from itertools import combinations





class ClassicCorrelation():
    
    def __init__(self, rets, method):
        
        self.rets = rets
        
        methods = ['pearson', 'spearman', 'kendall']
        
        if method not in methods:
            raise ValueError(f'method can only be one of {methods}, found {method}')
        
        self.method = method
        
    def sample_correlation(self):
        
        rets = self.rets.copy()
        
        corr = rets.corr(self.method)
        mask = np.triu(np.ones_like(corr, dtype=bool))
        
        f, ax = plt.subplots(figsize=(11, 9))
        
        ax = sns.heatmap(corr, mask = mask, annot = True, annot_kws={"size": 18})
        
    def rolling_correlation(self, window):
        
        rets = self.rets.copy()

        pairs = list(combinations(rets.columns.tolist(),2))
        
        corr = []
        
        for pair in pairs:
            
            correlation = rets[pair[0]].rolling(window=window).corr(rets[pair[1]], self.method).dropna()
            corr.append(correlation)
            
        self.corr_dataset = pd.concat([result for result in corr], axis = 1)
        self.corr_dataset.columns = pairs
        
        self.corr_dataset.plot(subplots=True, figsize = (20,20), grid = True, sharex=False)
            
        
        
class DynamicConditionalCorrelation():
    
    def __init__(self, rets):
        
        self.rets = rets
        self.on_data()
        self.fit()
        
        
    def fit_garch(self, series):
    
        model_garch = arch_model(series, dist = 't', rescale = False)
        fitted_garch = model_garch.fit(disp=0)

        return fitted_garch

    def prepare_data(self, series):
    
        fitted_garch = self.fit_garch(series)

        estimated_residual = fitted_garch.std_resid

        h = fitted_garch.conditional_volatility

        standardized_residual = estimated_residual / np.sqrt(h)

        cumulative_dist = t.cdf(standardized_residual, fitted_garch.params['nu'])

        return cumulative_dist
    
    def on_data(self):
    
        tickers = self.rets.columns

        residual_data = []

        for ticker in tickers:

            residual_data.append(self.prepare_data(self.rets[str(ticker)]))

        self.residual_data = np.array(residual_data)


    def initialize_Q(self):

        T = self.residual_data.shape[1]
        N = self.residual_data.shape[0]
        
        resid = self.residual_data

        sum = np.zeros([N,N])

        for i in range(T):

            sum += np.outer(resid[:,i], resid[:,i])

        return sum/T


    def Q_matrix(self, theta):

        Q_init = self.initialize_Q()
        
        residuals = self.residual_data
        
        Qt = [Q_init]

        a = theta[0] 
        b = theta[1]

        T = residuals.shape[1] - 1

        for i in range(T):

            eps_t = residuals[:, T-i]

            qt_1 = Qt[0]

            qt = (1.0 - a - b) * Q_init + a * np.outer(eps_t, eps_t) + b * qt_1

            Qt = [qt] + Qt

        return Qt
    
    def dcc_model(self, theta):


        Qt = self.Q_matrix(theta)

        N = Qt[0].shape[0]

        R_matrix = []

        for sub_matrix in Qt:

            std = 1.0/np.sqrt(np.abs(np.diag(sub_matrix))) * np.eye(N)

            R_matrix.append(np.dot(np.dot(std, sub_matrix), std))

        R_stack = []

        for i in range(len(R_matrix)):
            matrix = np.tril(R_matrix[i], k = -1)
            to_vec = np.matrix(matrix).A1
            R_stack.append(to_vec[to_vec !=0])

        return R_matrix, R_stack
        
        
    def loss_function(self, theta, arguments):

        Rt = self.dcc_model(theta)[0]
        arguments = self.residual_data
        eps_t = arguments

        loss = 0.0

        for i in range(len(Rt)):

            Ri = Rt[i]
            Ri_ = np.linalg.inv(Ri)

            et = eps_t[:,i]

            loss += np.log(np.linalg.det(Ri)) + np.dot(et, np.dot(Ri_, et)) - np.outer(et,et)

        return -0.5*np.sum(loss)

    
    def model_optimizer(self):

        cons = ({'type': 'ineq', 'fun': lambda x:  -x[0]  -x[1] +1})
        bnds = ((0, 0.5), (0, 0.9997))
        
        arguments = self.residual_data
        
        opt_out = minimize(self.loss_function, 
                           [0.01, 0.95], 
                           args = (arguments,), 
                           bounds=bnds, 
                           constraints=cons)

        return opt_out.x                            
        
        

    def fit(self):

        arguments = self.residual_data
        
        theta_optimal = self.model_optimizer()

        R, R_stack = self.dcc_model(theta_optimal)
        
        self.dcc_dataset = pd.DataFrame(data = R_stack, 
                                 index = self.rets.index,
                                 columns = list(combinations(self.rets.columns.tolist(), 2)))
        
        
        
    
    def plot_dcc(self):
        
        self.fit()
        
        df = self.dcc_dataset
        
        df.plot(subplots=True, figsize = (18,9), grid = True, sharex=False, logy=True)


        



        
        
        
        