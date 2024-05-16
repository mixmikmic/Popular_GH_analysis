import warnings
warnings.filterwarnings('ignore')

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

# let's split into 70/30: train=70% and test=30%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= .4, random_state = 0)

print("X train shape: ", X_train.shape)
print("X test shape: ", X_test.shape)
print()
print("y train shape: ", y_train.shape)
print("y test shape: ", y_test.shape)

# we'll set it to some parameters, but we'll go through depth on parameter tuning later
model = SVC(kernel='linear', C=1)
# fit our training data
model.fit(X_train, y_train)
#let's predict 
pred = model.predict(X_test)

#accuracy score
from sklearn.metrics import accuracy_score
accuracy_score(y_test, pred)

# let's get our classification report

#classification report
from sklearn.metrics import classification_report
print(classification_report(y_test, pred))

#confusion matrix
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, pred))





