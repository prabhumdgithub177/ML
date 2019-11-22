#Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('auto-mpg.csv')
X = dataset.iloc[:,0:-2].values
y = dataset.iloc[:,-1].values.reshape(-1,1)

print(dataset.shape)
print(dataset.head(10))
print(dataset.describe())

# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, [3]])
X[:, [3]] = imputer.transform(X[:, [3]])

'''import statsmodels.formula.api as sm
X=np.append(arr = np.ones((397,1)).astype(int),values=X,axis=1)
X_opt=X[:,[0,1,2,3,4,5,6,7]] # taking all datasets in X matrix
regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()
X_opt=X[:,[0,1,2,3,4,5,7]]
regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()'''

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 0)

# Fitting Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
lin_reg = RandomForestRegressor(n_estimators = 10, n_jobs=-1,random_state = 0)
lin_reg.fit(X_train, y_train)

# Predicting the Test set results
y_pred = lin_reg.predict(X_test).round()

#predicting result on multile feature
y_newpred= lin_reg.predict([[23,7,360,248,5450,5.5]])
