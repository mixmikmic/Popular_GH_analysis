# libraries we need
import numpy as np
from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC

# load iris dataset
data_holder = load_iris()
print(data_holder.data.shape)
print(data_holder.target.shape)

# set our X and y to data and target values
X , y = data_holder.data, data_holder.target

# split our data into train and test sets
# let's split into 70/30: train=70% and test=30%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= .4, random_state = 0)

print("X train shape: ", X_train.shape)
print("X test shape: ", X_test.shape)
print()
print("y train shape: ", y_train.shape)
print("y test shape: ", y_test.shape)

# let's fit into our model, svc
# we'll set it to some parameters, but we'll go through depth on parameter tuning later
model = SVC(kernel='linear', C=1)
# fit our training data
model.fit(X_train, y_train)
# print how our model is doing
print("Score: ", model.score(X_test, y_test))

# call cross-validation library
from sklearn.model_selection import cross_val_score
model = SVC(kernel='linear', C=1)

# let's try it using cv
scores = cross_val_score(model, X, y, cv=5)

scores

# print mean score
print("Accuracy using CV: ", scores.mean())



