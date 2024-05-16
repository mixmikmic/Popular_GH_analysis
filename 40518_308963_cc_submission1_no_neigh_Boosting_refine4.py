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
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier
from scipy.signal import medfilt
seed = 123
np.random.seed(seed)

import sys, scipy, sklearn
#print('Python:  ' + sys.version.split('\n')[0])
#print('         ' + sys.version.split('\n')[1])
print('Pandas:  ' + pd.__version__)
print('Numpy:   ' + np.__version__)
print('Scipy:   ' + scipy.__version__)
print('Sklearn: ' + sklearn.__version__)

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

reg = RandomForestRegressor(max_features='auto', n_estimators=250)
DataImpAll = data[feature_names].copy()
DataImp = DataImpAll.dropna(axis = 0, inplace=False)
Ximp=DataImp.loc[:, DataImp.columns != 'PE']
Yimp=DataImp.loc[:, 'PE']
reg.fit(Ximp, Yimp)
X[np.array(DataImpAll.PE.isnull()),4] = reg.predict(DataImpAll.loc[DataImpAll.PE.isnull(),:].drop('PE',axis=1,inplace=False))
data['PE']=X[:,4]

def add_more_features(data,feature_names):
    data['GR025']=np.power(np.abs(data['GR']),0.25)
    data['GR2']=np.power(np.abs(data['GR']),2)
    data['PHIND025']=np.power(np.abs(data['PHIND']),0.25)
    data['PHIND2']=np.power(np.abs(data['PHIND']),2)
    data['DeltaPHIlog']=np.power(data['DeltaPHI'],2)
    data['DeltaPHI05']=np.power(data['DeltaPHI'],3)
    data['NM_M_GR']= data['NM_M']* data['GR']
    data['NM_M_PHIND']= data['NM_M']* data['PHIND']
    data['NM_M_DeltaPHI']= data['NM_M']* data['DeltaPHI']
    data['GR_PHIND']= data['GR']* data['PHIND']
    data['NM_M_PE']= data['NM_M']* data['PE']
    data['NM_M_PE_GR']= data['NM_M']* data['PE']* data['GR']
    data['NM_M_PE_GR_PHIND']= data['NM_M']* data['PE']* data['GR']* data['PHIND']
    data['PE_GR_PHIND']= data['PE']* data['GR']* data['PHIND']
    data['PE_GR_PHIND_DeltaPHI']= data['PE']* data['GR']* data['PHIND']* data['DeltaPHI']
    feature_names= feature_names+['GR025','GR2','PHIND025','PHIND2','DeltaPHIlog','DeltaPHI05','NM_M_GR','NM_M_PHIND',
                                  'NM_M_DeltaPHI','GR_PHIND','NM_M_PE','NM_M_PE_GR','NM_M_PE_GR_PHIND','PE_GR_PHIND','PE_GR_PHIND_DeltaPHI']
    # Store features and labels
    X = data[feature_names].values  # features
    y = data['Facies'].values  # labels
    # Store well labels and depths
    well = data['Well Name'].values
    depth = data['Depth'].values
    return (data,feature_names,X,y,well,depth)

data,feature_names,X,y,well,depth= add_more_features(data,feature_names)

feature_names

data.isnull().sum()

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
    Norigfeat=len(feature_names)
    # Augment features
    X_aug = np.zeros((X.shape[0], X.shape[1]*(N_neig*2+2)))
    for w in np.unique(well):
        w_idx = np.where(well == w)[0]
        X_aug_win = augment_features_window(X[w_idx, :], N_neig)
        X_aug_grad = augment_features_gradient(X[w_idx, :], depth[w_idx])
        X_aug[w_idx, :] = np.concatenate((X_aug_win, X_aug_grad), axis=1)
       
    # Find padded rows
    padded_rows = np.unique(np.where(X_aug[:, 0:Norigfeat] == np.zeros((1, Norigfeat)))[0])
    
    return X_aug, padded_rows

# Augment features
X_aug, padded_rows = augment_features(X, well, depth, N_neig=0)

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

print('No of Feats',X.shape[1])

# Parameters search grid (uncomment parameters for full grid search... may take a lot of time)
N_grid = [150]  
MD_grid = [3]  
M_grid = [30] #[25,30]
LR_grid = [0.1]  
L_grid = [5]
S_grid = [20]#[10,20]  
param_grid = []
for N in N_grid:
    for M in MD_grid:
        for M1 in M_grid:
            for S in LR_grid: 
                for L in L_grid:
                    for S1 in S_grid:
                        param_grid.append({'N':N, 'MD':M, 'MF':M1,'LR':S,'L':L,'S1':S1})

# Train and test a classifier
def train_and_test(X_tr, y_tr, X_v, well_v, param):
    
    # Feature normalization
    scaler = preprocessing.RobustScaler(quantile_range=(25.0, 75.0)).fit(X_tr)
    X_tr = scaler.transform(X_tr)
    X_v = scaler.transform(X_v)
    
    # Train classifier
    #clf = OneVsOneClassifier(RandomForestClassifier(n_estimators=param['N'], criterion='entropy',
    #                         max_features=param['M'], min_samples_split=param['S'], min_samples_leaf=param['L'],
    #                         class_weight='balanced', random_state=0), n_jobs=-1)
    #clf = RandomForestClassifier(n_estimators=param['N'], criterion='entropy',
    #                         max_features=param['M'], min_samples_split=param['S'], min_samples_leaf=param['L'],
    #                         class_weight='balanced', random_state=0)
    #clf = OneVsOneClassifier(RandomForestClassifier(n_estimators=param['N'], criterion='gini',
    #                         max_features=param['M'], min_samples_split=param['S'], min_samples_leaf=param['L'],
    #                         class_weight='balanced', random_state=0), n_jobs=-1)
    # Train classifier  
    clf = OneVsOneClassifier(GradientBoostingClassifier(loss='exponential',
                                                        n_estimators=param['N'], 
                                                        learning_rate=param['LR'], 
                                                        max_depth=param['MD'],
                                                        max_features= param['MF'],
                                                        min_samples_leaf=param['L'],
                                                        min_samples_split=param['S1'],
                                                        random_state=seed, 
                                                        max_leaf_nodes=None, 
                                                        verbose=1), n_jobs=-1)
    
    clf.fit(X_tr, y_tr)
    
    # Test classifier
    y_v_hat = clf.predict(X_v)
    
    # Clean isolated facies for each well
    for w in np.unique(well_v):
        y_v_hat[well_v==w] = medfilt(y_v_hat[well_v==w], kernel_size=5)
    
    return (y_v_hat, clf)

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
        
        addnoise=1
        if ( addnoise==1 ):
            X_tr=X_tr+np.random.normal(loc=np.zeros(X_tr.shape[1]), scale=0.01*np.sqrt(np.std(X_tr,axis=0)/len(X_tr)), size=X_tr.shape)
        
        # Select well labels for validation data
        well_v = well[split['val']]

        # Train and test
        (y_v_hat,clf) = train_and_test(X_tr, y_tr, X_v, well_v, param)
        
        # Score
        score = f1_score(y_v, y_v_hat, average='micro')
        score_split.append(score)
        
    # Average score for this param
    score_param.append(np.mean(score_split))
    print('F1 score = %.3f %s' % (score_param[-1], param))
          
# Best set of parameters
best_idx = np.argmax(score_param)
param_best = param_grid[best_idx]
score_best = score_param[best_idx]
print('\nBest F1 score = %.3f %s' % (score_best, param_best))

# Load data from file
test_data = pd.read_csv('../validation_data_nofacies.csv')

feature_names_original = ['GR', 'ILD_log10', 'DeltaPHI', 'PHIND', 'PE', 'NM_M', 'RELPOS']
data,feature_names,X,y,well,depth= add_more_features(data,feature_names_original)

# Prepare training data
X_tr = X
y_tr = y

# Augment features
X_tr, padded_rows = augment_features(X_tr, well, depth)

# Removed padded rows
X_tr = np.delete(X_tr, padded_rows, axis=0)
y_tr = np.delete(y_tr, padded_rows, axis=0) 

test_data['Facies']=0
# Prepare test data
test_data,feature_names,X_ts,y_ts,well_ts,depth_ts= add_more_features(test_data,feature_names_original)
# Augment features
X_ts, padded_rows = augment_features(X_ts, well_ts, depth_ts)

# Predict test labels
y_ts_hat = train_and_test(X_tr, y_tr, X_ts, well_ts, param_best)

# Save predicted labels
test_data['Facies'] = y_ts_hat[0]
test_data.to_csv('cc_predicted_facies_noneigh_boosting_refine4_win.csv')

# Plot predicted labels
make_facies_log_plot(
    test_data[test_data['Well Name'] == 'STUART'],
    facies_colors=facies_colors)

make_facies_log_plot(
    test_data[test_data['Well Name'] == 'CRAWFORD'],
    facies_colors=facies_colors)
mpl.rcParams.update(inline_rc)



