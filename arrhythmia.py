#classification

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import scatter_matrix

# Importing the dataset
dataset = pd.read_csv('arrhythmia.csv')
#X = dataset.iloc[:, [0,1,3]].values # to choose only particular columns
#X = dataset.iloc[:,:-1].values  # to choose all the columns excluding final
X = dataset.iloc[:,0:-1].values
y = dataset.iloc[:,-1].values

print(dataset.shape)
print(dataset.head(5))
print(dataset.describe())

'''# class distribution
print(dataset.groupby('class').size())
# box and whisker plots
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()
# histograms
dataset.hist()
plt.show()
# scatter plot matrix
scatter_matrix(dataset)
plt.show()'''

# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:,[10]])
X[:,[10]] = imputer.transform(X[:,[10]])
imputer = imputer.fit(X[:,[11]])
X[:,[11]] = imputer.transform(X[:,[11]])
imputer = imputer.fit(X[:,[12]])
X[:,[12]] = imputer.transform(X[:,[12]])
imputer = imputer.fit(X[:,[13]])
X[:,[13]] = imputer.transform(X[:,[13]])
imputer = imputer.fit(X[:,[14]])
X[:,[14]] = imputer.transform(X[:,[14]])

#feature selection
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
bestfeatures = SelectKBest(score_func=chi2, k=10)
fit = bestfeatures.fit(X,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
#concat two dataframes for better visualization 
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['features','Score']  #naming the dataframe columns
print(featureScores.nlargest(100,'Score'))  #print 10 best features

#feature importance
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
model = ExtraTreesClassifier()
model.fit(X,y)
print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
#plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(4).plot(kind='barh')
plt.show()

# Encoding categorical data
# Encoding the Independent Variable
'''from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#onehotencoder is used to create dummy variables for categorical data
labelencoder_X = LabelEncoder()
X[:,33] = labelencoder_X.fit_transform(X[:,33])
onehotencoder = OneHotEncoder(categorical_features = [33])
X = onehotencoder.fit_transform(X).toarray()
X=X[:,1:]
# Encoding the Dependent Variable
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)'''

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
sc_y = StandardScaler()

from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='auto')))
models.append(('LDA', LinearDiscriminantAnalysis(n_components=20)))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
#models.append(('SVM', SVC(kernel = 'linear', random_state = 0)))
models.append(('SVM', SVC(kernel='rbf',gamma=0.71,random_state = 0)))
#models.append(('SVM', SVC(kernel = 'poly', degree=3, gamma=2, coef0=0.1)))
# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=0)
	cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)


# graph Comparison Algorithms 
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

# Make predictions on validation dataset
lda = LogisticRegression()
lda.fit(X_train, y_train)
y_pred = lda.predict(X_test)
accuracy=accuracy_score(y_test, y_pred)
print('accuracy:',accuracy)
cm = confusion_matrix(y_test, y_pred)
print('cm:','\n', cm)
cr=classification_report(y_test, y_pred)
print('report:','\n',cr)



