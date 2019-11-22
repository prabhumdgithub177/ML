# Wine quality detection
#classification
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Wine.csv')
# dataset after feature selection
X = dataset.iloc[:,[1,3,4,6,9,12]].values
y = dataset.iloc[:,-1].values

print(dataset.shape)
print(dataset.head(7))
print(dataset.describe())

'''from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
bestfeatures = SelectKBest(score_func=chi2, k=6)
fit = bestfeatures.fit(X,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
#concat two dataframes for better visualization 
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['features','Score']  #naming the dataframe columns
print(featureScores.nlargest(6,'Score'))  #print 10 best features


from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
model = ExtraTreesClassifier()
model.fit(X,y)
print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
#plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(6).plot(kind='barh')
plt.show()'''


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


# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis(n_components=2)))
models.append(('KNN', KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)))
models.append(('CART', DecisionTreeClassifier(criterion = 'entropy', random_state = 0,splitter='best',max_depth=None,min_samples_split=2)))
models.append(('NB', GaussianNB()))
#models.append(('SVM', SVC(kernel = 'linear')))
models.append(('SVM', SVC(kernel='rbf', C=1,gamma=0.71,max_iter=-1)))
#models.append(('SVM', SVC(kernel = 'poly', degree=3, gamma=2, coef0=1,tol=0.001)))
#models.append(('Random',RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)))
models.append(('RandomP',RandomForestClassifier()))
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
lda = SVC()
lda.fit(X_train, y_train)
y_pred = lda.predict(X_test)
accuracy=accuracy_score(y_test, y_pred)
print('accuracy:',accuracy)
cm = confusion_matrix(y_test, y_pred)
print('cm:','\n', cm)
cr=classification_report(y_test, y_pred)
print('report:','\n',cr)


#making prediction on new input data
y_newpred=lda.predict(sc.transform([[6,1,4,1,2,1]]))
