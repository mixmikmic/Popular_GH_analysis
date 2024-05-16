import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

from pandas import set_option
set_option("display.max_rows", 10)

from sklearn import preprocessing

filename = '../facies_vectors.csv'
train = pd.read_csv(filename)

# encode well name and formation features
le = preprocessing.LabelEncoder()
train["Well Name"] = le.fit_transform(train["Well Name"])
train["Formation"] = le.fit_transform(train["Formation"])

data_loaded = train.copy()

# cleanup memory
del train

data_loaded

from sklearn import preprocessing

data = data_loaded.copy()

impPE_features = ['Facies', 'Formation', 'Well Name', 'GR', 'ILD_log10', 'DeltaPHI', 'PHIND', 'NM_M', 'RELPOS']
rmse = []

for w in data["Well Name"].unique():
    wTrain = data[(data["PE"].notnull()) & (data["Well Name"] != w)]
    wTest = data[(data["PE"].notnull()) & (data["Well Name"] == w)]
    
    if wTest.shape[0] > 0:
        yTest = wTest["PE"].values
        
        meanPE = wTrain["PE"].mean()
        wTest["predictedPE"] = meanPE
        
        rmse.append((((yTest - wTest["predictedPE"])**2).mean())**0.5)
        
print(rmse)
print("Average RMSE:" + str(sum(rmse)/len(rmse)))

# cleanup memory
del data

from sklearn.ensemble import RandomForestRegressor

data = data_loaded.copy()

impPE_features = ['Facies', 'Formation', 'Well Name', 'GR', 'ILD_log10', 'DeltaPHI', 'PHIND', 'NM_M', 'RELPOS']
rf = RandomForestRegressor(max_features='sqrt', n_estimators=100, random_state=1)
rmse = []

for w in data["Well Name"].unique():
    wTrain = data[(data["PE"].isnull() == False) & (data["Well Name"] != w)]
    wTest = data[(data["PE"].isnull() == False) & (data["Well Name"] == w)]
    
    if wTest.shape[0] > 0:
        XTrain = wTrain[impPE_features].values
        yTrain = wTrain["PE"].values
        XTest = wTest[impPE_features].values
        yTest = wTest["PE"].values
        
        w_rf = rf.fit(XTrain, yTrain)
        
        predictedPE = w_rf.predict(XTest)
        rmse.append((((yTest - predictedPE)**2).mean())**0.5)
    
print(rmse)
print("Average RMSE:" + str(sum(rmse)/len(rmse)))

# cleanup memory
del data

data = data_loaded.copy()

rf_train = data[data['PE'].notnull()]
rf_test = data[data['PE'].isnull()]

xTrain = rf_train[impPE_features].values
yTrain = rf_train["PE"].values
xTest = rf_test[impPE_features].values

rf_fit = rf.fit(xTrain, yTrain)
predictedPE = rf_fit.predict(xTest)
data["PE"][data["PE"].isnull()] = predictedPE

data_imputed = data.copy()

# cleanup memory
del data

# output
data_imputed

facies_labels = ['SS','CSiS','FSiS','SiSh','MS','WS','D','PS','BS']

data = data_imputed.copy()

features = ["GR", "ILD_log10", "DeltaPHI", "PHIND", "PE"]
for f in features:
    facies_mean = data[f].groupby(data["Facies"]).mean()
    
    for i in range(0, len(facies_mean)):
        data[f + "_" + facies_labels[i] + "_SqDev"] = (data[f] - facies_mean.values[i])**2

data_fe = data.copy()

del data
data_fe

# Feature windows concatenation function
def augment_features_window(X, N_neig):
    
    # Parameters
    N_row = X.shape[0]
    N_feat = X.shape[1]

    # Zero padding
    X = np.vstack((np.zeros((N_neig, N_feat)), X, (np.zeros((N_neig, N_feat)))))

    # Loop over windows
    X_aug = np.zeros((N_row, N_feat*(2*N_neig+1)))
    for r in np.arange(N_row)+N_neig:
        this_row = []
        for c in np.arange(-N_neig,N_neig+1):
            this_row = np.hstack((this_row, X[r+c]))
        X_aug[r-N_neig] = this_row

    return X_aug

# Feature gradient computation function
def augment_features_gradient(X, depth):
    
    # Compute features gradient
    d_diff = np.diff(depth).reshape((-1, 1))
    d_diff[d_diff==0] = 0.001
    X_diff = np.diff(X, axis=0)
    X_grad = X_diff / d_diff
        
    # Compensate for last missing value
    X_grad = np.concatenate((X_grad, np.zeros((1, X_grad.shape[1]))))
    
    return X_grad

# Feature augmentation function
def augment_features(X, well, depth, N_neig=1):
    
    # Augment features
    X_aug = np.zeros((X.shape[0], X.shape[1]*(N_neig*2+2)))
    for w in np.unique(well):
        w_idx = np.where(well == w)[0]
        X_aug_win = augment_features_window(X[w_idx, :], N_neig)
        X_aug_grad = augment_features_gradient(X[w_idx, :], depth[w_idx])
        X_aug[w_idx, :] = np.concatenate((X_aug_win, X_aug_grad), axis=1)
    
    # Find padded rows
    padded_rows = np.unique(np.where(X_aug[:, 0:7] == np.zeros((1, 7)))[0])
    
    return X_aug, padded_rows

data = data_fe.copy()

remFeatures = ["Facies", "Well Name", "Depth"]
x = list(data)
features = [f for f in x if f not in remFeatures]

X = data[features].values
y = data["Facies"].values

# Store well labels and depths
well = data['Well Name']
depth = data['Depth'].values

X_aug, padded_rows = augment_features(X, well.values, depth)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
#from classification_utilities import display_cm, display_adj_cm

# 1) loops through wells - splitting data (current well held out as CV/test)
# 2) trains model (using all wells excluding current)
# 3) evaluates predictions against known values and adds f1-score to array
# 4) returns average f1-score (expected f1-score)
def cvTrain(X, y, well, params):
    rf = RandomForestClassifier(max_features=params['M'], n_estimators=params['N'], criterion='entropy', 
                                min_samples_split=params['S'], min_samples_leaf=params['L'], random_state=1)
    f1 = []
    
    for w in well.unique():
        Xtrain_w = X[well.values != w]
        ytrain_w = y[well.values != w]
        Xtest_w = X[well.values == w]
        ytest_w = y[well.values == w]
        
        w_rf = rf.fit(Xtrain_w, ytrain_w)
        predictedFacies = w_rf.predict(Xtest_w)
        f1.append(f1_score(ytest_w, predictedFacies, average='micro'))
        
    f1 = (sum(f1)/len(f1))
    return f1

# parameters search grid (uncomment for full grid search - will take a long time)
N_grid = [250]    #[50, 250, 500]        # n_estimators
M_grid = [75]     #[25, 50, 75]          # max_features
S_grid = [5]      #[5, 10]               # min_samples_split
L_grid = [2]      #[2, 3, 5]             # min_samples_leaf

# build grid of hyperparameters
param_grid = []
for N in N_grid:
    for M in M_grid:
        for S in S_grid:
            for L in L_grid:
                param_grid.append({'N':N, 'M':M, 'S':S, 'L':L})
                
# loop through parameters and cross-validate models for each
for params in param_grid:
    print(str(params) + ' Average F1-score: ' + str(cvTrain(X_aug, y, well, params)))

from sklearn import preprocessing

filename = '../validation_data_nofacies.csv'
test = pd.read_csv(filename)

# encode well name and formation features
le = preprocessing.LabelEncoder()
test["Well Name"] = le.fit_transform(test["Well Name"])
test["Formation"] = le.fit_transform(test["Formation"])
test_loaded = test.copy()

facies_labels = ['SS','CSiS','FSiS','SiSh','MS','WS','D','PS','BS']

train = data_imputed.copy()

features = ["GR", "ILD_log10", "DeltaPHI", "PHIND", "PE"]
for f in features:
    facies_mean = train[f].groupby(train["Facies"]).mean()
    
    for i in range(0, len(facies_mean)):
        test[f + "_" + facies_labels[i] + "_SqDev"] = (test[f] - facies_mean.values[i])**2

test_fe = test.copy()

del test

test_fe

test = test_fe.copy()

remFeatures = ["Well Name", "Depth"]
x = list(test)
features = [f for f in x if f not in remFeatures]

Xtest = test[features].values

# Store well labels and depths
welltest = test['Well Name']
depthtest = test['Depth'].values

Xtest_aug, test_padded_rows = augment_features(Xtest, welltest.values, depthtest)

from sklearn.ensemble import RandomForestClassifier

test = test_loaded.copy()

rf = RandomForestClassifier(max_features=75, n_estimators=250, criterion='entropy', 
                                min_samples_split=5, min_samples_leaf=2, random_state=1)
fit = rf.fit(X_aug, y)
predictedFacies = fit.predict(Xtest_aug)

test["Facies"] = predictedFacies
test.to_csv('jpoirier011_submission001.csv')

