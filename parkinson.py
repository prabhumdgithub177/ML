# classification
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import warnings
from pandas.plotting import scatter_matrix
warnings.simplefilter('ignore')
sns.set()

#loading dataset
from sklearn.datasets import load_iris
dataset=load_iris()


# Importing the dataset
dataset = pd.read_csv('parkinsonsdata.csv')
#finding outliers
from scipy import stats
z=np.abs(stats.zscore(dataset.iloc[:,1:]))
print(z)
threshold=3
print(np.where(z>3))
print(z[73][1])

q1=dataset.quantile(0.25)
q3=dataset.quantile(0.75)
IQR=q3-q1
print(IQR)

print((dataset<(q1-1.5*IQR))|(dataset>(q3+1.5*IQR)))
dataset=dataset[(z<3).all(axis=1)]
dataset.shape
dataset.size

X=dataset.iloc[:,[1,2,3,10,14,16]].values
y=dataset.iloc[:,-1].values

features=dataset.loc[:,dataset.columns!='status'].values[:,1:]
target=dataset.loc[:,'status'].values

print(dataset.columns)
print(dataset.shape)
print(dataset.head(7))
print(dataset.tail(7))
print(dataset.describe())
print(X.shape)
print(y[y==1].shape[0],y[y==0].shape[0])
dataset.corr(method='pearson')


'''from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
bestfeatures = SelectKBest(score_func=chi2, k=12)
fit = bestfeatures.fit(X,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
#concat two dataframes for better visualization 
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['features','Score']  #naming the dataframe columns
print(featureScores.nlargest(12,'Score'))  #print 10 best features

from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
model = ExtraTreesClassifier()
model.fit(X,y)
print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
#plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(12).plot(kind='barh')
plt.show()'''

sns.heatmap(X)



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


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
sc = StandardScaler()
sc = MinMaxScaler((-1,1))
X=sc.fit_transform(X_train,X_test)
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
models.append(('CART', DecisionTreeClassifier(criterion = 'gini', splitter='best',min_samples_split=2)))
models.append(('NB', GaussianNB(var_smoothing=1e-9)))
#models.append(('SVM', SVC(kernel = 'linear', random_state = 0)))
#models.append(('SVM', SVC(kernel = 'poly', degree=3, gamma=2, coef0=1)))
#models.append(('Random',RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 5,bootstrap=True,max_depth=70)))
models.append(('Random',RandomForestClassifier(n_estimators = 10,random_state = 0)))
models.append(('SVM', SVC(kernel='rbf',gamma=0.71,random_state = 0,C=1)))
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
lda = SVC()
lda.fit(X_train, y_train)
y_pred = lda.predict(X_test)
#y_pred = lda.score(X_test,y_test)
accuracy=accuracy_score(y_test, y_pred)
print('accuracy:',accuracy)
cm = confusion_matrix(y_test, y_pred)
print('cm:','\n', cm)
cr=classification_report(y_test, y_pred)
print('report:','\n',cr)


Pcorrect=(y_test == y_pred).sum()
Pwrong=(y_test != y_pred).sum()
Accuracy=Pcorrect/39
Error=Pwrong/39     

'''from sklearn.model_selection import GridSearchCV
parameters={'n_estimators':  [10], 
            'max_features':[3], 
            'random_state' : [5],
            'bootstrap':[True]}

grid_search=GridSearchCV(estimator=lda,
                         param_grid=parameters,
                         cv=10,n_jobs=-1,verbose=2,
                         scoring = 'accuracy',
                         refit=True)
grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_'''


cor=dataset.corr()
fig=plt.figure()
ax=fig.add_subplot(111)
cax=ax.matshow(cor,vmin=-1,vmax=1)
fig.colorbar(cax)
