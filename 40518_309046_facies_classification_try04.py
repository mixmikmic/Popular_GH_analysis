# Import
from __future__ import division
get_ipython().magic('matplotlib inline')
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['figure.figsize'] = (20.0, 10.0)
inline_rc = dict(mpl.rcParams)
from classification_utilities import make_facies_log_plot

import pandas as pd
import numpy as np
import seaborn as sns

from sklearn import preprocessing
from sklearn.model_selection import LeavePGroupsOut
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsOneClassifier

from scipy.signal import medfilt

from xgboost import XGBClassifier

import sys, scipy, sklearn, xgboost
print('Python:  ' + sys.version.split('\n')[0])
print('         ' + sys.version.split('\n')[1])
print('Pandas:  ' + pd.__version__)
print('Numpy:   ' + np.__version__)
print('Scipy:   ' + scipy.__version__)
print('Sklearn: ' + sklearn.__version__)
print('XGBoost: ' + xgboost.__version__)

# Parameters
feature_names = ['GR', 'ILD_log10', 'DeltaPHI', 'PHIND', 'PE', 'NM_M', 'RELPOS']
facies_names = ['SS', 'CSiS', 'FSiS', 'SiSh', 'MS', 'WS', 'D', 'PS', 'BS']
facies_colors = ['#F4D03F', '#F5B041','#DC7633','#6E2C00', '#1B4F72','#2E86C1', '#AED6F1', '#A569BD', '#196F3D']

# Load data from file
data = pd.read_csv('../facies_vectors.csv')

# Store features and labels
X = data[feature_names].values  # features
y = data['Facies'].values  # labels

# Store well labels and depths
well = data['Well Name'].values
depth = data['Depth'].values

# Sort data according to depth for each well
for w_idx, w in enumerate(np.unique(well)):
    X_well = X[well == w]
    X[well == w] = X_well[np.argsort(depth[well == w])]
    depth[well == w] = np.sort(depth[well == w])

# Define function for plotting feature statistics
def plot_feature_stats(X, y, feature_names, facies_colors, facies_names):
    
    # Remove NaN
    nan_idx = np.any(np.isnan(X), axis=1)
    X = X[np.logical_not(nan_idx), :]
    y = y[np.logical_not(nan_idx)]
    
    # Merge features and labels into a single DataFrame
    features = pd.DataFrame(X, columns=feature_names)
    labels = pd.DataFrame(y, columns=['Facies'])
    for f_idx, facies in enumerate(facies_names):
        labels[labels[:] == f_idx] = facies
    data = pd.concat((labels, features), axis=1)

    # Plot features statistics
    facies_color_map = {}
    for ind, label in enumerate(facies_names):
        facies_color_map[label] = facies_colors[ind]

    sns.pairplot(data, hue='Facies', palette=facies_color_map, hue_order=list(reversed(facies_names)))

# Feature distribution
plot_feature_stats(X, y, feature_names, facies_colors, facies_names)
mpl.rcParams.update(inline_rc)

# Facies per well
for w_idx, w in enumerate(np.unique(well)):
    ax = plt.subplot(3, 4, w_idx+1)
    hist = np.histogram(y[well == w], bins=np.arange(len(facies_names)+1)+.5)
    plt.bar(np.arange(len(hist[0])), hist[0], color=facies_colors, align='center')
    ax.set_xticks(np.arange(len(hist[0])))
    ax.set_xticklabels(facies_names)
    ax.set_title(w)

# Features per well
for w_idx, w in enumerate(np.unique(well)):
    ax = plt.subplot(3, 4, w_idx+1)
    hist = np.logical_not(np.any(np.isnan(X[well == w, :]), axis=0))
    plt.bar(np.arange(len(hist)), hist, color=facies_colors, align='center')
    ax.set_xticks(np.arange(len(hist)))
    ax.set_xticklabels(feature_names)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['miss', 'hit'])
    ax.set_title(w)

imp = preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0)
imp.fit(X)
X = imp.transform(X)

# Feature gradient computation function
def augment_features_gradient(X, depth):
    
    # Compute features gradient in both directions
    d_diff = np.diff(depth).reshape((-1, 1))
    d_diff[d_diff==0] = 0.001
    X_diff = np.diff(X, axis=0)
    X_grad_aux = X_diff / d_diff    
    
    # Compensate for last missing value
    X_grad = np.concatenate((X_grad_aux, np.zeros((1, X_grad_aux.shape[1]))))
    
    return X_grad

# Feature augmentation function
def augment_features(X, well, depth):
    
    # Augment features
    padded_rows = []
    X_aug = np.zeros((X.shape[0], X.shape[1]*2))
    for w in np.unique(well):
        w_idx = np.where(well == w)[0]
        X_aug_grad = augment_features_gradient(X[w_idx, :], depth[w_idx])
        X_aug[w_idx, :] = np.concatenate((X[w_idx, :], X_aug_grad), axis=1)
        padded_rows.append(w_idx[-1])
        
    # Find padded rows
    #padded_rows = np.where(~X_aug[:, 7:].any(axis=1))[0]
    
    return X_aug, padded_rows

# Augment features
X_aug, padded_rows = augment_features(X, well, depth)

# Add higher degree terms and interaction terms to the model
deg = 2
poly = preprocessing.PolynomialFeatures(deg, interaction_only=False)
X_aug = poly.fit_transform(X_aug)
X_aug = X_aug[:,1:]

# Initialize model selection methods
lpgo = LeavePGroupsOut(2)

# Generate splits
split_list = []
for train, val in lpgo.split(X, y, groups=data['Well Name']):
    hist_tr = np.histogram(y[train], bins=np.arange(len(facies_names)+1)+.5)
    hist_val = np.histogram(y[val], bins=np.arange(len(facies_names)+1)+.5)
    if np.all(hist_tr[0] != 0) & np.all(hist_val[0] != 0):
        split_list.append({'train':train, 'val':val})
            
# Print splits
for s, split in enumerate(split_list):
    print('Split %d' % s)
    print('    training:   %s' % (data['Well Name'][split['train']].unique()))
    print('    validation: %s' % (data['Well Name'][split['val']].unique()))

# Parameters search grid (uncomment parameters for full grid search... may take a lot of time)
N_grid = [100] # [50, 100, 150, 200]
D_grid = [3] # [2, 3, 5, 7]
W_grid = [8] # [6, 7, 8, 10, 12]
R_grid = [.12] # [.10, .11, .12, .13, .14]
T_grid = [.9] # [.8, .9, .10]
MED_grid = [1] # [0, 1]
param_grid = []
for N in N_grid:
    for D in D_grid:
        for W in W_grid:
            for R in R_grid:
                for T in T_grid:
                    for MED in MED_grid:
                        param_grid.append({'N':N, 'D':D, 'W':W, 'R':R, 'T':T, 'MED':MED})

# Train and test a classifier
def train_and_test(X_tr, y_tr, X_v, well_v, param):
    
    # Feature normalization
    scaler = preprocessing.RobustScaler(quantile_range=(25.0, 75.0)).fit(X_tr)
    X_tr = scaler.transform(X_tr)
    X_v = scaler.transform(X_v)
    
    # Prepare layers
    idx_2a = np.where(y_tr < 4)[0]
    idx_2b = np.where(y_tr >= 4)[0]
    y_tr_1 = y_tr.copy()
    y_tr_1[idx_2a] = 0
    y_tr_1[idx_2b] = 1
    
    # Define classifiers
    clf_2a = XGBClassifier(learning_rate=param['R'], max_depth=param['D'], min_child_weight=param['W'],
                           n_estimators=param['N'], seed=0, colsample_bytree=param['T'])
    clf_2b = XGBClassifier(learning_rate=param['R'], max_depth=param['D'], min_child_weight=param['W'],
                           n_estimators=param['N'], seed=0, colsample_bytree=param['T'])

    # Train layers
    clf_2a.fit(X_tr[idx_2a], y_tr[idx_2a])
    clf_2b.fit(X_tr[idx_2b], y_tr[idx_2b])
    
    # Test classifier
    y_v_hat_1 = np.zeros((X_v.shape[0]))
    y_v_hat_1[np.where(X_v[:, 5] <= np.mean(X_v[:, 5]))] = 0
    y_v_hat_1[np.where(X_v[:, 5] > np.mean(X_v[:, 5]))] = 1
    idx_2a_hat = np.where(y_v_hat_1 == 0)[0]
    idx_2b_hat = np.where(y_v_hat_1 == 1)[0]
    y_v_hat_2a = clf_2a.predict(X_v[idx_2a_hat])
    y_v_hat_2b = clf_2b.predict(X_v[idx_2b_hat])
    y_v_hat = y_v_hat_1.copy()
    y_v_hat[idx_2a_hat] = y_v_hat_2a
    y_v_hat[idx_2b_hat] = y_v_hat_2b
    
    # Clean isolated facies for each well
    if param['MED']:
        for w in np.unique(well_v):
            y_v_hat[well_v==w] = medfilt(y_v_hat[well_v==w], kernel_size=5)
    
    return y_v_hat

# For each set of parameters
score_param = []
for param in param_grid:
    
    # For each data split
    score_split = []
    for split in split_list:
    
        # Remove padded rows
        split_train_no_pad = np.setdiff1d(split['train'], padded_rows)
        
        # Select training and validation data from current split
        X_tr = X_aug[split_train_no_pad, :]
        X_v = X_aug[split['val'], :]
        y_tr = y[split_train_no_pad]
        y_v = y[split['val']]
        
        # Select well labels for validation data
        well_v = well[split['val']]

        # Train and test
        y_v_hat = train_and_test(X_tr, y_tr, X_v, well_v, param)
        
        # Score
        score = f1_score(y_v, y_v_hat, average='micro')
        score_split.append(score)
        
        # Score per classifier
        y_v_1 = y_v.copy()
        y_v_1[np.where(y_v < 4)[0]] = 0
        y_v_1[np.where(y_v >= 4)[0]] = 1
        y_v_hat_1 = y_v_hat.copy()
        y_v_hat_1[np.where(y_v_hat_1 < 4)[0]] = 0
        y_v_hat_1[np.where(y_v_hat_1 >= 4)[0]] = 1
        score_1 = f1_score(y_v_1, y_v_hat_1, average='micro')
        score_2a = f1_score(y_v[np.where(y_v < 4)[0]], y_v_hat[np.where(y_v < 4)[0]], average='micro')
        score_2b = f1_score(y_v[np.where(y_v >= 4)[0]], y_v_hat[np.where(y_v >= 4)[0]], average='micro')
        print('   F1 score split = %.3f (%.3f, %.3f, %.3f)' % (score_split[-1], score_1, score_2a, score_2b)) 
        
    # Average score for this param
    score_param.append(np.mean(score_split))
    print('F1 score = %.3f (std %.3f) %s' % (score_param[-1], np.std(score_split), param))
          
# Best set of parameters
best_idx = np.argmax(score_param)
param_best = param_grid[best_idx]
score_best = score_param[best_idx]
print('\nBest F1 score = %.3f %s' % (score_best, param_best))

# Load data from file
test_data = pd.read_csv('../validation_data_nofacies.csv')

# Prepare training data
X_tr = X
y_tr = y

# Augment features
X_tr, padded_rows = augment_features(X_tr, well, depth)

# Removed padded rows
X_tr = np.delete(X_tr, padded_rows, axis=0)
y_tr = np.delete(y_tr, padded_rows, axis=0)

# Prepare test data
well_ts = test_data['Well Name'].values
depth_ts = test_data['Depth'].values
X_ts = test_data[feature_names].values

# Sort data according to depth for each well
for w_idx, w in enumerate(np.unique(well_ts)):
    X_ts_well = X_ts[well_ts == w]
    X_ts[well_ts == w] = X_ts_well[np.argsort(depth_ts[well_ts == w])]
    depth_ts[well_ts == w] = np.sort(depth_ts[well_ts == w])

# Augment features
X_ts, padded_rows = augment_features(X_ts, well_ts, depth_ts)

# Predict test labels
y_ts_hat = train_and_test(X_tr, y_tr, X_ts, well_ts, param_best)

# Save predicted labels
test_data['Facies'] = y_ts_hat
test_data.to_csv('well_data_with_facies_try04.csv')

# Plot predicted labels
make_facies_log_plot(
    test_data[test_data['Well Name'] == 'STUART'],
    facies_colors=facies_colors)

make_facies_log_plot(
    test_data[test_data['Well Name'] == 'CRAWFORD'],
    facies_colors=facies_colors)
mpl.rcParams.update(inline_rc)

