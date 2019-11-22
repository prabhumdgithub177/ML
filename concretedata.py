# concrete strength prediction

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing the dataset
dataset = pd.read_csv('concretedata.csv')
X = dataset.iloc[:, [0,1,2,3,7]].values #after backward elimination
y = dataset.iloc[:, -1].values.reshape(-1,1)


# Backward elimination
'''import statsmodels.formula.api as sm
X=np.append(arr = np.ones((1030,1)).astype(int),values=X,axis=1)
X_new=X[:,[0,1,2,3,4,5,6,7,8]] # taking all datasets in X matrix
# eliminate larger p values step by step
regressor_OLS=sm.OLS(endog=y,exog=X_new).fit()
regressor_OLS.summary()
X_new=X[:,[0,1,2,3,4,5,8]]
regressor_OLS=sm.OLS(endog=y,exog=X_new).fit()
regressor_OLS.summary()
X_new=X[:,[0,1,2,3,4,8]]
regressor_OLS=sm.OLS(endog=y,exog=X_new).fit()
regressor_OLS.summary()'''


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Fitting Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test).round(2)


# to predict result for new data
y_newpred = regressor.predict([[280,76,0,125,80]])


