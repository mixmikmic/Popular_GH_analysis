get_ipython().magic('matplotlib inline')
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.metrics import f1_score, accuracy_score, make_scorer

filename = 'engineered_features.csv'
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

from sklearn.ensemble import RandomForestClassifier

from mlxtend.feature_selection import SequentialFeatureSelector as SFS
clf = RandomForestClassifier(random_state=49)

sfs = SFS(clf, 
          k_features=100, 
          forward=True, 
          floating=False, 
          scoring=Fscorer,
          cv = 8,
          n_jobs = -1)

sfs = sfs.fit(X, y)

np.save('sfs_RF_metric_dict.npy', sfs.get_metric_dict()) 

# load previously saved dictionary
read_dictionary = np.load('sfs_RF_metric_dict.npy').item()

# plot results
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs

# run this twice
fig = plt.figure()                                                               
ax = plot_sfs(read_dictionary, kind='std_err')
fig_size = plt.rcParams["figure.figsize"] 
fig_size[0] = 22
fig_size[1] = 18

plt.title('Sequential Forward Selection (w. StdDev)')
plt.grid()
plt.xticks( rotation='vertical')
locs, labels = plt.xticks()
plt.xticks( locs, labels)
plt.show()

# save results to dataframe
selected_summary = pd.DataFrame.from_dict(read_dictionary).T
selected_summary['index'] = selected_summary.index
selected_summary.sort_values(by='avg_score', ascending=0)

# save dataframe
selected_summary.to_csv('SFS_RF_selected_features_summary.csv', sep=',', header=True, index = False)

# re load saved dataframe and sort by score
filename = 'SFS_RF_selected_features_summary.csv'
selected_summary = pd.read_csv(filename)
selected_summary = selected_summary.set_index(['index'])
selected_summary.sort_values(by='avg_score', ascending=0).head()

# feature selection with highest score
selected_summary.iloc[44]['feature_idx']

slct = np.array([257, 3, 4, 6, 7, 8, 10, 12, 16, 273, 146, 19, 26, 27, 284, 285, 30, 34, 163, 1, 42, 179, 155, 181, 184, 58, 315, 190, 320, 193, 194, 203, 290, 80, 210, 35, 84, 90, 97, 18, 241, 372, 119, 120, 126])
slct

# isolate and save selected features
filename = 'engineered_features_validation_set2.csv'
training_data = pd.read_csv(filename)
X = training_data.drop(['Formation', 'Well Name'], axis=1)
Xs = X.iloc[:, slct]
Xs = pd.concat([training_data[['Depth', 'Well Name', 'Formation']], Xs], axis = 1)
print np.shape(Xs), list(Xs)

Xs.to_csv('SFS_top45_selected_engineered_features_validation_set.csv', sep=',',  index=False)

# feature selection with highest score
selected_summary.iloc[74]['feature_idx']

slct = np.array([257, 3, 4, 5, 6, 7, 8, 265, 10, 12, 13, 16, 273, 18, 19, 26, 27, 284, 285, 30, 34, 35, 1, 42, 304, 309, 313, 58, 315, 319, 320, 75, 80, 338, 84, 341, 89, 90, 92, 97, 101, 102, 110, 372, 119, 120, 122, 124, 126, 127, 138, 139, 146, 155, 163, 165, 167, 171, 177, 179, 180, 181, 184, 190, 193, 194, 198, 203, 290, 210, 211, 225, 241, 249, 253])
slct

# isolate and save selected features
filename = 'engineered_features_validation_set2.csv'
training_data = pd.read_csv(filename)
X = training_data.drop(['Formation', 'Well Name'], axis=1)
Xs = X.iloc[:, slct]
Xs = pd.concat([training_data[['Depth', 'Well Name', 'Formation']], Xs], axis = 1)
print np.shape(Xs), list(Xs)

Xs.to_csv('SFS_top75_selected_engineered_features_validation_set.csv', sep=',',  index=False)



