#classification of skin diseases 

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('dermatology1.csv')
X = dataset.iloc[:,0:33].values
y = dataset.iloc[:,34].values

print(dataset.shape)
print(dataset.head(10))
print(dataset.describe())


# Finding missing data
'''from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, :33])
X[:, :33] = imputer.transform(X[:, :33])'''

# Encoding categorical data of indpnt var.
'''from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#onehotencoder is used to create dummy variables for categorical data
labelencoder_X = LabelEncoder()
X[:,33] = labelencoder_X.fit_transform(X[:,33])
onehotencoder = OneHotEncoder(categorical_features = [33])
X = onehotencoder.fit_transform(X).toarray()
X=X[:,1:]# dummy variable trap
# Encoding the Dependent Variable
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)'''

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.137, random_state = 0)

# Feature Scaling
'''from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
sc_y = StandardScaler()'''

from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import KernelPCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis(n_components=2)))
models.append(('KNN', KNeighborsClassifier()))
#models.append(('KPCA', KernelPCA(n_components=2, kernel='rbf')))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(kernel = 'linear')))
#models.append(('SVM', SVC(kernel='rbf',gamma= 0.71)))
#models.append(('SVM', SVC(kernel = 'poly', degree=3, gamma=2, coef0=0.01)))
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
lda = KNeighborsClassifier()# 100 % accuracy
lda.fit(X_train, y_train)
y_pred = lda.predict(X_test)
accuracy=accuracy_score(y_test, y_pred)
print('accuracy:',accuracy)
cm = confusion_matrix(y_test, y_pred)
print('cm:','\n', cm)
cr=classification_report(y_test, y_pred)
print('report:','\n',cr)



