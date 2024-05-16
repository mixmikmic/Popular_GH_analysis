import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm

iris = datasets.load_iris()
iris.data.shape, iris.target.shape

get_ipython().magic('pinfo train_test_split')

X_train, X_test, y_train, y_test = train_test_split(
     iris.data, iris.target, test_size=0.4, random_state=0)

X_train.shape, y_train.shape

X_test.shape, y_test.shape

clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
clf.score(X_test, y_test)  

clf.score(X_train, y_train)

from sklearn.model_selection import cross_val_score

get_ipython().magic('pinfo cross_val_score')

clf = svm.SVC(kernel='linear', C=1)

scores = cross_val_score(clf, iris.data, iris.target, cv=2)

scores

print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

from sklearn import metrics

scores = cross_val_score(
     clf, iris.data, iris.target, cv=5, scoring='f1_macro')

scores

from sklearn.model_selection import ShuffleSplit

get_ipython().magic('pinfo ShuffleSplit')

n_samples = iris.data.shape[0]

cv = ShuffleSplit(n_splits=3, test_size=0.3, random_state=0)

cross_val_score(clf, iris.data, iris.target, cv=cv)

from sklearn.model_selection import cross_val_predict

get_ipython().magic('pinfo cross_val_predict')

predicted = cross_val_predict(clf, iris.data, iris.target, cv=10)

predicted.shape

metrics.accuracy_score(iris.target, predicted) 

from sklearn.linear_model import LassoCV

from sklearn.model_selection import KFold

get_ipython().magic('pinfo KFold')

kf = KFold(n_splits=4, shuffle=True)

X = ["a", "b", "c", "d"]
for train, test in kf.split(X):
    print("%s %s" % (train, test))

from sklearn.model_selection import StratifiedKFold

get_ipython().magic('pinfo StratifiedKFold')

X = np.ones(10)
y = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
skf = StratifiedKFold(n_splits=3)
for train, test in skf.split(X, y):
    print("%s %s" % (train, test))

from sklearn.model_selection import GroupKFold

X = [0.1, 0.2, 2.2, 2.4, 2.3, 4.55, 5.8, 8.8, 9, 10]
y = ["a", "b", "b", "b", "c", "c", "c", "d", "d", "d"]
groups = [1, 1, 1, 2, 2, 2, 3, 3, 3, 3]

gkf = GroupKFold(n_splits=3)
for train, test in gkf.split(X, y, groups=groups):
    print("%s %s" % (train, test))

from sklearn.model_selection import TimeSeriesSplit

X = np.array([[1, 2], [3, 4], [1, 2], [3, 4], [1, 2], [3, 4]])
y = np.array([1, 2, 3, 4, 5, 6])
tscv = TimeSeriesSplit(n_splits=3)
print(tscv)  

for train, test in tscv.split(X):
    print("%s %s" % (train, test))



