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
selected_summary.iloc[39]['feature_idx']

slct = np.array([256, 257, 3, 6, 1, 264, 137, 23, 280, 281, 288, 289, 113, 168, 7, 304, 305, 312, 193, 328, 
                 329, 224, 80, 81, 83, 122, 95, 352, 353, 232, 233, 295, 208, 109, 336, 360, 118, 248, 250, 255])
slct

# isolate and save selected features
filename = 'engineered_features.csv'
training_data = pd.read_csv(filename)
X = training_data.drop(['Formation', 'Well Name','Facies'], axis=1)
Xs = X.iloc[:, slct]
Xs = pd.concat([training_data[['Depth', 'Well Name', 'Formation', 'Facies']], Xs], axis = 1)
print np.shape(Xs), list(Xs)

Xs.to_csv('SFS_top40_selected_engineered_features.csv', sep=',',  index=False)

# feature selection with highest score
selected_summary.iloc[69]['feature_idx']

slct = np.array([256, 257, 3, 4, 6, 1, 264, 9, 17, 277, 23, 280, 281, 283, 288, 289, 295, 40, 7, 304, 305, 308, 265, 
                 312, 317, 360, 97, 328, 329, 331, 79, 80, 81, 83, 89, 350, 95, 352, 353, 99, 104, 364, 109, 113, 
                 118, 120, 122, 128, 137, 149, 151, 153, 168, 169, 171, 174, 193, 196, 207, 208, 224, 336, 226, 
                 227, 232, 233, 25, 248, 250, 255])
slct

# isolate and save selected features
filename = 'engineered_features.csv'
training_data = pd.read_csv(filename)
X = training_data.drop(['Formation', 'Well Name','Facies'], axis=1)
Xs = X.iloc[:, slct]
Xs = pd.concat([training_data[['Depth', 'Well Name', 'Formation', 'Facies']], Xs], axis = 1)
print np.shape(Xs), list(Xs)

Xs.to_csv('SFS_top70_selected_engineered_features.csv', sep=',',  index=False)



