# Abalone age prediction
# Regression
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('abalone.csv')
X = dataset.iloc[:,  [3,4,6]].values
#X = dataset.iloc[:, 3].values.reshape(-1,1)
y = dataset.iloc[:, -1].values.reshape(-1,1)

#feature selection
'''from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
bestfeatures = SelectKBest(score_func=chi2, k=4)
fit = bestfeatures.fit(X,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
#concat two dataframes for better visualization 
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['features','Score']  #naming the dataframe columns
print(featureScores.nlargest(4,'Score'))  #print 10 best features

#feature importance
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
model = ExtraTreesClassifier()
model.fit(X,y)
print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
#plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(4).plot(kind='barh')
plt.show()'''


# Backward elimination also gives the same features
'''import statsmodels.formula.api as sm
X=np.append(arr = np.ones((4177,1)).astype(int),values=X,axis=1)
X_new=X[:,[0,1,2,3,4,5,6]] # taking all datasets in X matrix
# eliminate larger p values step by step
regressor_OLS=sm.OLS(endog=y,exog=X_new).fit()
regressor_OLS.summary()
X_new=X[:,[3,4,5,6]]
regressor_OLS=sm.OLS(endog=y,exog=X_new).fit()
regressor_OLS.summary()'''


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Fitting Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 5)
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test).round()

#predicting result on single feature
y_newpred = regressor.predict([[0.05]]).round()

#predicting result on multile feature
y_newpred= regressor.predict([[0.157,0.918,0.245]])
