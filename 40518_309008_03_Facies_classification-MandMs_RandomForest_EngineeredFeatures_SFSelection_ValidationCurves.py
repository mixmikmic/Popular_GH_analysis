get_ipython().magic('matplotlib inline')
import numpy as np
import scipy as sp
from scipy.stats import randint as sp_randint
from scipy.signal import argrelextrema
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

from sklearn import preprocessing
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import LeaveOneGroupOut, validation_curve

filename = 'SFS_top70_selected_engineered_features.csv'
training_data = pd.read_csv(filename)
training_data.describe()

training_data['Well Name'] = training_data['Well Name'].astype('category')
training_data['Formation'] = training_data['Formation'].astype('category')
training_data['Well Name'].unique()

y = training_data['Facies'].values
print y[25:40]
print np.shape(y)

X = training_data.drop(['Formation', 'Well Name','Facies'], axis=1)
print np.shape(X)
X.describe(percentiles=[.05, .25, .50, .75, .95])

scaler = preprocessing.StandardScaler().fit(X)
X = scaler.transform(X)

Fscorer = make_scorer(f1_score, average = 'micro')

wells = training_data["Well Name"].values
logo = LeaveOneGroupOut()

from sklearn.ensemble import RandomForestClassifier
RF_clf100 = RandomForestClassifier (n_estimators=100, n_jobs=-1, random_state = 49)
RF_clf200 = RandomForestClassifier (n_estimators=200, n_jobs=-1, random_state = 49)
RF_clf300 = RandomForestClassifier (n_estimators=300, n_jobs=-1, random_state = 49)
RF_clf400 = RandomForestClassifier (n_estimators=400, n_jobs=-1, random_state = 49)
RF_clf500 = RandomForestClassifier (n_estimators=500, n_jobs=-1, random_state = 49)
RF_clf600 = RandomForestClassifier (n_estimators=600, n_jobs=-1, random_state = 49)

param_name = "max_features"
#param_range = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
param_range = [9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51]

plt.figure()
plt.suptitle('n_estimators = 100', fontsize=14, fontweight='bold')
_, test_scores = validation_curve(RF_clf100, X, y, cv=logo.split(X, y, groups=wells),
                                  param_name=param_name, param_range=param_range,
                                  scoring=Fscorer, n_jobs=-1)
test_scores_mean = np.mean(test_scores, axis=1)
plt.plot(param_range, test_scores_mean)
plt.xlabel(param_name)
plt.xlim(min(param_range), max(param_range))
plt.ylabel("F1")
plt.ylim(0.47, 0.57)
plt.show()
#print max(test_scores_mean[argrelextrema(test_scores_mean, np.greater)])
print np.amax(test_scores_mean) 
print np.array(param_range)[test_scores_mean.argmax(axis=0)] 

plt.figure()
plt.suptitle('n_estimators = 200', fontsize=14, fontweight='bold')
_, test_scores = validation_curve(RF_clf200, X, y, cv=logo.split(X, y, groups=wells),
                                  param_name=param_name, param_range=param_range,
                                  scoring=Fscorer, n_jobs=-1)
test_scores_mean = np.mean(test_scores, axis=1)
plt.plot(param_range, test_scores_mean)
plt.xlabel(param_name)
plt.xlim(min(param_range), max(param_range))
plt.ylabel("F1")
plt.ylim(0.47, 0.57)
plt.show()
#print max(test_scores_mean[argrelextrema(test_scores_mean, np.greater)])
print np.amax(test_scores_mean) 
print np.array(param_range)[test_scores_mean.argmax(axis=0)] 

plt.figure()
plt.suptitle('n_estimators = 300', fontsize=14, fontweight='bold')
_, test_scores  = validation_curve(RF_clf300, X, y, cv=logo.split(X, y, groups=wells),
                                  param_name=param_name, param_range=param_range,
                                  scoring=Fscorer, n_jobs=-1)
test_scores_mean = np.mean(test_scores, axis=1)
plt.plot(param_range, test_scores_mean)
plt.xlabel(param_name)
plt.xlim(min(param_range), max(param_range))
plt.ylabel("F1")
plt.ylim(0.47, 0.57)
plt.show() 
#print max(test_scores_mean[argrelextrema(test_scores_mean, np.greater)])
print np.amax(test_scores_mean) 
print np.array(param_range)[test_scores_mean.argmax(axis=0)] 

plt.figure()
plt.suptitle('n_estimators = 400', fontsize=14, fontweight='bold')
_, test_scores  = validation_curve(RF_clf400, X, y, cv=logo.split(X, y, groups=wells),
                                  param_name=param_name, param_range=param_range,
                                  scoring=Fscorer, n_jobs=-1)
test_scores_mean = np.mean(test_scores, axis=1)
plt.plot(param_range, test_scores_mean)
plt.xlabel(param_name)
plt.xlim(min(param_range), max(param_range))
plt.ylabel("F1")
plt.ylim(0.47, 0.57)
plt.show()
#print max(test_scores_mean[argrelextrema(test_scores_mean, np.greater)])
print np.amax(test_scores_mean) 
print np.array(param_range)[test_scores_mean.argmax(axis=0)] 

plt.figure()
plt.suptitle('n_estimators = 500', fontsize=14, fontweight='bold')
_, test_scores  = validation_curve(RF_clf500, X, y, cv=logo.split(X, y, groups=wells),
                                  param_name=param_name, param_range=param_range,
                                  scoring=Fscorer, n_jobs=-1)
test_scores_mean = np.mean(test_scores, axis=1)
plt.plot(param_range, test_scores_mean)
plt.xlabel(param_name)
plt.xlim(min(param_range), max(param_range))
plt.ylabel("F1")
plt.ylim(0.47, 0.57)
plt.show()
#print max(test_scores_mean[argrelextrema(test_scores_mean, np.greater)])
print np.amax(test_scores_mean) 
print np.array(param_range)[test_scores_mean.argmax(axis=0)] 

plt.figure()
plt.suptitle('n_estimators = 600', fontsize=14, fontweight='bold')
_, test_scores  = validation_curve(RF_clf600, X, y, cv=logo.split(X, y, groups=wells),
                                  param_name=param_name, param_range=param_range,
                                  scoring=Fscorer, n_jobs=-1)
test_scores_mean = np.mean(test_scores, axis=1)
plt.plot(param_range, test_scores_mean)
plt.xlabel(param_name)
plt.xlim(min(param_range), max(param_range))
plt.ylabel("F1")
plt.ylim(0.47, 0.57)
plt.show()
#print max(test_scores_mean[argrelextrema(test_scores_mean, np.greater)])
print np.amax(test_scores_mean) 
print np.array(param_range)[test_scores_mean.argmax(axis=0)] 

RF_clf_f1 = RandomForestClassifier (n_estimators=600, max_features = 21,
                                  n_jobs=-1, random_state = 49)

f1_RF = []

for train, test in logo.split(X, y, groups=wells):
    well_name = wells[test[0]]
    RF_clf_f1.fit(X[train], y[train])
    pred = RF_clf_f1.predict(X[test])
    sc = f1_score(y[test], pred, labels = np.arange(10), average = 'micro')
    print("{:>20s}  {:.3f}".format(well_name, sc))
    f1_RF.append(sc)
    
print "-Average leave-one-well-out F1 Score: %6f" % (sum(f1_RF)/(1.0*(len(f1_RF))))

RF_clf_b = RandomForestClassifier (n_estimators=600, max_features = 21,
                                  n_jobs=-1, random_state = 49)

blind = pd.read_csv('engineered_features_validation_set_top70.csv') 
X_blind = np.array(blind.drop(['Formation', 'Well Name'], axis=1)) 
scaler1 = preprocessing.StandardScaler().fit(X_blind)
X_blind = scaler1.transform(X_blind) 
y_pred = RF_clf_b.fit(X, y).predict(X_blind) 
#blind['Facies'] = y_pred

np.save('ypred_RF_SFS_VC.npy', y_pred)

