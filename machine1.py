#

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing the dataset
dataset = pd.read_csv('machine1.csv')
X = dataset.iloc[:,  [3,4,5,7]].values
y = dataset.iloc[:, -2].values

print(dataset.shape)
print(dataset.head(5))
print(dataset.describe())

'''import statsmodels.formula.api as sm
X=np.append(arr = np.ones((208,1)).astype(int),values=X,axis=1)
X_opt=X[:,[0,1,2,3,4,5]] # taking all datasets in X matrix
# eliminate larger p values step by step
regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()
X_opt=X[:,[0,2,3,4,5]]
regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()'''


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 0)


# Fitting Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X, y)

# Predicting the Test set results
y_pred = (regressor.predict(X_test)).round()


# predicting result for new data
y_newpred = regressor.predict([[4000,16000,65,1]])




