import pandas as pd
df = pd.read_csv('forest-cover-type.csv')
df.head()

X, y = df.iloc[:, 1:-1].values, df.iloc[:, -1].values

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.6, random_state=0)

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score

models = [
    ('tree', DecisionTreeClassifier(random_state=0)),
    ('bagged tree', BaggingClassifier(
            DecisionTreeClassifier(random_state=0),
            random_state=0,
            n_estimators=10))
]

for label, model in models:  
    model.fit(X_train, y_train) 
    print("{} training|test accuracy: {:.2f} | {:.2f}".format(
            label, 
            accuracy_score(y_train, model.predict(X_train)),
            accuracy_score(y_test, model.predict(X_test))))    


# your code goes here!

# your code goes here!

from sklearn.tree import DecisionTreeClassifier
from sklearn.base import BaseEstimator
import numpy as np

class McBaggingClassifier(BaseEstimator):
    
    def __init__(self, classifier_factory=DecisionTreeClassifier, num_classifiers=10):
        self.classifier_factory = classifier_factory
        self.num_classifiers = num_classifiers
        
    def fit(self, X, y):
        # create num_classifier classifiers calling classifier_factory, each 
        # fitted with a different sample from X
        return self

    def predict(self, X):
        # get the prediction for each classifier, take a majority vote        
        return np.ones(X.shape[0])

our_models = [
    ('tree', DecisionTreeClassifier(random_state=0)),
    ('our bagged tree', McBaggingClassifier(
            classifier_factory=lambda: DecisionTreeClassifier(random_state=0)
            ))
]

for label, model in our_models:  
    model.fit(X_train, y_train) 
    print("{} training|test accuracy: {:.2f} | {:.2f}".format(
        label, 
        accuracy_score(y_train, model.predict(X_train)),
        accuracy_score(y_test, model.predict(X_test))))    

