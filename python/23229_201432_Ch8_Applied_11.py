import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

get_ipython().magic('matplotlib inline')

df = pd.read_csv('../../../data/Caravan.csv', index_col=0)

df.head()

# Define the predictors and the response variables
predictors = df.columns.tolist()
predictors.remove('Purchase')

X = df[predictors].values
y = df['Purchase'].values

# use the first 1000 as training and the remainder for testing
X_train = X[0:1000]
X_test = X[1000:]
y_train = y[0:1000]
y_test = y[1000:]

# build and fit a boosting model to the training data
booster = GradientBoostingClassifier(learning_rate=0.01, n_estimators=1000, max_depth=3, 
                                     random_state=0)
boost_est = booster.fit(X_train, y_train)

# get the variable importance
Importances = pd.DataFrame(boost_est.feature_importances_, index=predictors, 
             columns=['Importance']).sort_values(by='Importance', ascending=False)
Importances.head(8)

y_pred = boost_est.predict_proba(X_test)
print(y_pred)

# if the yes probability exceeds 0.2 then assign it as a purchase
pred_purchase = ['No'if row[1] < 0.2 else 'Yes' for row in y_pred ]

cm = confusion_matrix(y_true = y_test, y_pred=pred_purchase, labels=['No', 'Yes'])
print(cm)

# The CM matrix is [[NN NY]
#                   [YN, YY] 
# where C_ij is equal to the number of observations known to be in group i but 
# predicted to be in group j.

#so the fraction predicted to be Yes that are actually Yes is 
cm[1,1]/(cm[1,1]+cm[0,1])

# Build KNN clasifier
knn_est = KNeighborsClassifier(n_neighbors=5).fit(X_train,y_train)
# make predictons
predicted_class = knn_est.predict_proba(X_test)
# if the yes probability exceeds 0.2 then assign it as a purchase
knn_pred = ['No'if row[1] < 0.2 else 'Yes' for row in predicted_class ]
# build confusion matrix
cm = confusion_matrix(y_true = y_test, y_pred=knn_pred, labels=['No', 'Yes'])
print(cm)

#so the fraction predicted to be Yes that are actually Yes is 
cm[1,1]/(cm[1,1]+cm[0,1])



