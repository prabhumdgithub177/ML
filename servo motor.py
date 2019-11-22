#Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('servo.csv')
X = dataset.iloc[:,  0:-1].values
y = dataset.iloc[:, -1:].values

print(dataset.shape)
print(dataset.head(10))
print(dataset.describe())


# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#onehotencoder is used to create dummy variables for categorical data
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
X[:, 1] = labelencoder_X.fit_transform(X[:, 1])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()
X= X[:, 1:]
# Encoding the Dependent Variable
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 0)


# Fitting Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
lin_reg = RandomForestRegressor(n_estimators = 10, random_state = 0)
lin_reg.fit(X, y)

# Predicting the Test set results
y_pred = lin_reg.predict(X_test)

# predicting result for new data
y_newpred = lin_reg.predict([[1,2,0,3,1,4,0]])


