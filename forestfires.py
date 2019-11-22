
#Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing the dataset
dataset = pd.read_csv('forestfires.csv')
#selected features for best accuracy
X = dataset.iloc[:,  [0,1,6,7,9]].values
y = dataset.iloc[:, -1].values


print(dataset.shape)
print(dataset.head(5))
print(dataset.describe())


'''import statsmodels.formula.api as sm
X=np.append(arr = np.ones((517,1)).astype(int),values=X,axis=1)
X_opt=X[:,[0,1,2,3,4,5]] # taking all datasets in X matrix
# eliminate larger p values step by step
regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()
X_opt=X[:,[0,1,2,3,5]]
regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()'''


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 0)


# Fitting Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
lin_reg = RandomForestRegressor(n_estimators = 10, random_state = 0)
lin_reg.fit(X_train, y_train)


# Predicting the Test set results
y_pred = lin_reg.predict(X_test).round(2)



