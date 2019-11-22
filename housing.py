# Boston house price prediction
#Regression
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import scatter_matrix
import seaborn as sn

# Importing the dataset
dataset = pd.read_csv('housing.csv')

#finding outliers
from scipy import stats
z=np.abs(stats.zscore(dataset))
print(z)
threshold=3
print(np.where(z>3))
print(z[55][1])

q1=dataset.quantile(0.25)
q3=dataset.quantile(0.75)
IQR=q3-q1
print(IQR)

print((dataset<(q1-1.5*IQR))|(dataset>(q3+1.5*IQR)))
dataset=dataset[(z<3).all(axis=1)]
dataset.shape
dataset.size

# from backward elmimination
X = dataset.iloc[:,[0,1,4,5,6,7,8,9,12]].values 
y = dataset.iloc[:, -1].values.round()


#dataset description
print(dataset.shape)
print(dataset.size)
print(dataset.head(10))
print(dataset.describe())
print(dataset.columns)


#backward elimination
'''import statsmodels.formula.api as sm
X=np.append(arr = np.ones((506,1)).astype(int),values=X,axis=1)
X_opt=X[:,[0,1,2,3,4,5,6,7,8,9,10,11,12]] # taking all datasets in X matrix
regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()
X_opt=X[:,[0,1,4,5,6,7,8,9,11,12]]
regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()
X_opt=X[:,[0,2,4,5,6,11,12]]
regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()'''

sn.boxplot(X[9])

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Fitting Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor()
regressor.fit(X_train, y_train)


# Predicting the Test set results
y_pred = regressor.predict(X_test).round()


# Accuracy prediction
Pcorrect=(y_test == y_pred).sum()
Pwrong=(y_test != y_pred).sum()  
Accuracy=Pcorrect/83
Error=Pwrong/83









