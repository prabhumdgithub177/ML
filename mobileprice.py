#mobile price classification based on specification
     
    
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import scatter_matrix
import seaborn as sn


# Importing the dataset
dataset = pd.read_csv('mobileprice.csv')

#from feature selection
X = dataset.iloc[:, [13,11,0,12,8,6,15,16,4,14]].values
y = dataset.iloc[:,-1].values

plt.figure(figsize=(17,15))
cor=dataset.corr()
sn.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()

cor_target=abs(cor['price_range'])
imp_features=cor_target[cor_target>0.5]
imp_features


print(dataset[['battery_power','blue','clock_speed','dual_sim']].corr())
print(dataset.shape)
print(dataset.head(10))
print(X.shape)
print(X.size)
print(dataset.describe())


#feature selection
'''from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
bestfeatures = SelectKBest(score_func=chi2, k=10)
fit = bestfeatures.fit(X,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(dataset.columns)# coz X don't have columns error
#concat two dataframes for better visualization 
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  #naming the dataframe columns
print(featureScores.nlargest(10,'Score'))  #print 10 best features

#feature importance
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
model = ExtraTreesClassifier()
model.fit(X,y)
print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
#plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(12).plot(kind='barh')
plt.show()'''

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
'''from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)'''

#importing packages
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
models.append(('Random',RandomForestClassifier(n_estimators=5)))
#models.append(('SVM', SVC(kernel = 'linear', random_state = 0)))
#models.append(('SVM', SVC(kernel='rbf',gamma=0.71,coef0=0.1,decision_function_shape='ovr',tol=1e-3,degree=3)))
#models.append(('SVM', SVC(kernel = 'poly', degree=3, gamma=2, coef0=1)))
#models.append(('RandomP',RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state=5, n_jobs=-1,min_samples_split=2,min_samples_leaf=1)))
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
lda = KNeighborsClassifier()
lda.fit(X_train, y_train)
y_pred = lda.predict(X_test)
accuracy=accuracy_score(y_test, y_pred)
print('accuracy:',accuracy)
cm = confusion_matrix(y_test, y_pred)
print('cm:','\n', cm)
cr=classification_report(y_test, y_pred)
print('report:','\n',cr)

#Accuracy prediction
Pcorrect=(y_test == y_pred).sum().round(2)
Pwrong=(y_test != y_pred).sum()  
accuracy=Pcorrect/400
error=Pwrong/400

# predicting result on single input
'''y_newpred = lda.predict([[1,2,3,4,5,6..]])'''


#predicting result for multiple inputs
test_dataset = pd.read_csv('mobile testset.csv')
X_new = test_dataset.iloc[:, [13,11,0,12,8,6,15,16,4,14]].values
X_new.shape

x=test_dataset.iloc[:,[13,11,0,12,8,6,15,16,4,14]]

y_newpred = lda.predict(X_new)
#y_newpred = np.where(y_newpred>0.5, 1, 0)

 
x.append(y_newpred).to_csv('abc.csv')
#export data to csv file
import csv
with open("mobiletestprediction.csv",'w') as f:
    writer=csv.writer(f,lineterminator='\n')
    writer.writerow(['ram','px_height','battery_power','px_width','mobile_wt','int_memory','sc_w','talk_time','fc','sc_h','price'])# write columns name in excel
    for ram,px_height,battery_power,px_width,mobile_wt,int_memory,sc_w,talk_time,fc,sc_h,price in zip(x,y_newpred.astype(int)):
        writer.writerow([ram,px_height,battery_power,px_width,mobile_wt,int_memory,sc_w,talk_time,fc,sc_h,price])

        
