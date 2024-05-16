param_grid = [
  {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
  {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
 ]

from sklearn.model_selection import GridSearchCV

get_ipython().magic('pinfo GridSearchCV')

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.svm import SVC

digits = datasets.load_digits()

n_samples = len(digits.images)
X = digits.images.reshape((n_samples, -1))
y = digits.target

# Split the dataset in two equal parts
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=0)

tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]



clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=5,
                       scoring='f1_macro')
clf.fit(X_train, y_train)

clf.best_params_

clf.cv_results_

y_true, y_pred = y_test, clf.predict(X_test)
print classification_report(y_true, y_pred)

clf.cv_results_.keys()

for param, score in zip(clf.cv_results_['params'], clf.cv_results_['mean_test_score']):
    print param, score

import scipy

params = {'C': scipy.stats.expon(scale=100), 'gamma': scipy.stats.expon(scale=.1),
  'kernel': ['rbf'], 'class_weight':['balanced', None]}

from sklearn.model_selection import RandomizedSearchCV

get_ipython().magic('pinfo RandomizedSearchCV')

clf = RandomizedSearchCV(SVC(), params, cv=5,
                       scoring='f1_macro')
clf.fit(X_train, y_train)

clf.best_params_

clf.cv_results_

y_true, y_pred = y_test, clf.predict(X_test)
print classification_report(y_true, y_pred)

for param, score in zip(clf.cv_results_['params'], clf.cv_results_['mean_test_score']):
    print param, score



