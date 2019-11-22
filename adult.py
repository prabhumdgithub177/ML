#predicting if person earns more than 50k or lesser PA in a given census data
#classification
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing the dataset
dataset = pd.read_csv('adult1.csv')
#X= dataset.iloc[:,[0,1,2,4,10,11,12,13]].values
#X = dataset.iloc[:,[0,2,4,10,11,12]].values
y = dataset.iloc[:,0:14].values
y = dataset.iloc[:,14].values


#apply SelectKBest class to extract top 10 best features
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
bestfeatures = SelectKBest(score_func=chi2, k=5)
fit = bestfeatures.fit(X,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)

#concat two dataframes for better visualization 
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['features','Score']  #naming the dataframe columns
print(featureScores.nlargest(5,'Score'))  #print 10 best features

#feature importance graph
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
model = ExtraTreesClassifier()
model.fit(X,y)
print(model.feature_importances_) 
#use inbuilt class feature_importances of tree based classifiers
#plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(4).plot(kind='barh')
plt.show()


print(dataset.shape)
print(dataset.head(100))
print(dataset.describe())
'


# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:,[1]])
X[:,[1]] = imputer.transform(X[:,[1]])
imputer = imputer.fit(X[:,[6]])
X[:,[6]] = imputer.transform(X[:,[6]])
imputer = imputer.fit(X[:,[13]])
X[:,[13]] = imputer.transform(X[:,[13]])

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#onehotencoder is used to create dummy variables for categorical data
labelencoder_X = LabelEncoder()
X[:,1] = labelencoder_X.fit_transform(X[:,1])
X[:,3] = labelencoder_X.fit_transform(X[:,3])
X[:,5] = labelencoder_X.fit_transform(X[:,5])
X[:,6] = labelencoder_X.fit_transform(X[:,6])
X[:,7] = labelencoder_X.fit_transform(X[:,7])
X[:,8] = labelencoder_X.fit_transform(X[:,8])
X[:,9] = labelencoder_X.fit_transform(X[:,9])
X[:,13] = labelencoder_X.fit_transform(X[:,13])
onehotencoder = OneHotEncoder(categorical_features = [6])
X = onehotencoder.fit_transform(X).toarray()
X=X[:,1:]
# Encoding the Dependent Variable
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


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
from sklearn.ensemble import RandomForestClassifier


# Spot Check Algorithms    apply only SVC
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis(n_components=2)))
models.append(('KNN', KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)))
models.append(('CART', DecisionTreeClassifier(criterion = 'entropy', random_state = 0)))
models.append(('NB', GaussianNB()))
models.append(('Random',RandomForestClassifier()))
#models.append(('Random',RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)))
#models.append(('SVM', SVC(kernel = 'linear', random_state = 0)))
models.append(('SVM', SVC(kernel='rbf',gamma=0.71,random_state = 0)))
#models.append(('SVM', SVC(kernel = 'poly', degree=3, gamma=2, coef0=1)))
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


# Make predictions on validation dataset
var = SVC()# 84% accuracy
var.fit(X_train, y_train)
y_pred = var.predict(X_test)
accuracy=accuracy_score(y_test, y_pred)
print('accuracy:',accuracy)
cm = confusion_matrix(y_test, y_pred)
print('cm:','\n', cm)
cr=classification_report(y_test, y_pred)
print('report:','\n',cr)



