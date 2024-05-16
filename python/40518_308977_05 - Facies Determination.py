import pandas as pd
import bokeh.plotting as bk
import numpy as np

from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from tpot import TPOTClassifier, TPOTRegressor

import sys
sys.path.append('~/home/slygeorge/Documents/Python/SEG ML Competition')

from IPython.core.display import display, HTML
display(HTML("<style>.container { width:80% !important; }</style>"))

bk.output_notebook()

# Input file paths
train_path = '../training_data.csv'

# Read training data to dataframe
train = pd.read_csv(train_path)

# TPOT library requires that the target class is renamed to 'class'
train.rename(columns={'Facies': 'class'}, inplace=True)

well_names = train['Well Name']

facies_labels = ['SS', 'CSiS', 'FSiS', 'SiSh', 'MS', 'WS', 'D', 'PS', 'BS']

train.head()

train.dropna().describe()

# Some quick-look plots, PE has been highlighted, as this appears to be missing from the alternative version of the training dataset
plots = []
for well, group in train.groupby('Well Name'):
    group = group.sort_values(by = 'Depth')
    plots.append(bk.figure(height = 500, width = 150))
    plots[-1].line(group['PE'], group['Depth'], color = 'blue')
    plots[-1].line(group['DeltaPHI'], group['Depth'], color = 'red')
    plots[-1].title.text = well
    
grid = bk.gridplot([plots])
bk.show(grid)

# Set string features to integers

for i, value in enumerate(train['Formation'].unique()):
    train.loc[train['Formation'] == value, 'Formation'] = i
    
for i, value in enumerate(train['Well Name'].unique()):
    train.loc[train['Well Name'] == value, 'Well Name'] = i

# Used to reassign index, initally after attempting to upsample results

train['orig_index'] = train.index

# Define resample factors
resample_factors = [2, 5, 10, 25, 50]

initial_columns = ['Formation', 'Well Name', 'Depth', 'GR', 'ILD_log10',
       'DeltaPHI', 'PHIND', 'PE', 'NM_M', 'RELPOS']
div_columns = ['Depth', 'GR', 'ILD_log10',
       'DeltaPHI', 'PHIND', 'PE', 'NM_M', 'RELPOS']

# Use rolling windows through upsampled frame, grouping by well name.

# Empty list to hold frames
mean_frames = []
above = []
below = []

for well, group in train.groupby('Well Name'):
    # Empty list to hold rolling frames
    constructor_list = []
    for f in resample_factors:
        
        working_frame = group[['Depth', 'GR', 'ILD_log10', 'DeltaPHI', 'PHIND', 'PE', 'NM_M',
       'RELPOS']]
        
        mean_frame = working_frame.rolling(window = f, center = True).mean().interpolate(method = 'index', limit_direction = 'both', limit = None)
        mean_frame.columns = ['Mean_{0}_{1}'.format(f, column) for column in mean_frame.columns]
        max_frame = working_frame.rolling(window = f, center = True).max().interpolate(method = 'index', limit_direction = 'both', limit = None)
        max_frame.columns = ['Max_{0}_{1}'.format(f, column) for column in max_frame.columns]
        min_frame = working_frame.rolling(window = f, center = True).min().interpolate(method = 'index', limit_direction = 'both', limit = None)
        min_frame.columns = ['Min_{0}_{1}'.format(f, column) for column in min_frame.columns]
        std_frame = working_frame.rolling(window = f, center = True).std().interpolate(method = 'index', limit_direction = 'both', limit = None)
        std_frame.columns = ['Std_{0}_{1}'.format(f, column) for column in std_frame.columns]
        var_frame = working_frame.rolling(window = f, center = True).var().interpolate(method = 'index', limit_direction = 'both', limit = None)
        var_frame.columns = ['Var_{0}_{1}'.format(f, column) for column in var_frame.columns]
        diff_frame = working_frame.diff(f, axis = 0).interpolate(method = 'index', limit_direction = 'both', limit = None)
        diff_frame.columns = ['Diff_{0}_{1}'.format(f, column) for column in diff_frame.columns]
        rdiff_frame = working_frame.sort_index(ascending = False).diff(f, axis = 0).interpolate(method = 'index', limit_direction = 'both', limit = None).sort_index()
        rdiff_frame.columns = ['Rdiff_{0}_{1}'.format(f, column) for column in rdiff_frame.columns]
        skew_frame = working_frame.rolling(window = f, center = True).skew().interpolate(method = 'index', limit_direction = 'both', limit = None)
        skew_frame.columns = ['Skew_{0}_{1}'.format(f, column) for column in skew_frame.columns]
        
        f_frame = pd.concat((mean_frame, max_frame, min_frame, std_frame, var_frame, diff_frame, rdiff_frame), axis = 1)
        
        constructor_list.append(f_frame)
        
    well_frame = pd.concat(constructor_list, axis = 1)
    well_frame['class'] = group['class']
    well_frame['Well Name'] = well
    # orig index is holding the original index locations, to make extracting the results trivial
    well_frame['orig_index'] = group['orig_index']
    df = group.sort_values('Depth')
    u = df.shift(-1).fillna(method = 'ffill')
    b = df.shift(1).fillna(method = 'bfill')
    above.append(u[div_columns])
    below.append(b[div_columns])
    
    mean_frames.append(well_frame.fillna(method = 'bfill').fillna(method = 'ffill'))

# Concatenate all sub-frames together into single 'upsampled_frane'
frame = train
frame.index = frame['orig_index']
frame.drop(['orig_index', 'class', 'Well Name'], axis = 1, inplace = True)

for f in mean_frames:
    f.index = f['orig_index']

rolling_frame = pd.concat(mean_frames, axis = 0)
above_frame = pd.concat(above)
above_frame.columns = ['above_'+ column for column in above_frame.columns]
below_frame = pd.concat(below)
below_frame.columns = ['below_'+ column for column in below_frame.columns]
upsampled_frame = pd.concat((frame, rolling_frame, above_frame, below_frame), axis = 1)

# Features is the column set used for training the model
features = [feature for feature in upsampled_frame.columns if 'class' not in feature]

# Normalise dataset
std_scaler = preprocessing.StandardScaler().fit(upsampled_frame[features])

train_std = std_scaler.transform(upsampled_frame[features])

train_std_frame = upsampled_frame
for i, column in enumerate(features):
    train_std_frame.loc[:, column] = train_std[:, i]

upsampled_frame_std = train_std_frame

# Create ratios between features
div_columns = ['Depth', 'GR', 'ILD_log10',
       'DeltaPHI', 'PHIND', 'PE', 'NM_M', 'RELPOS']

for feature in div_columns:
    for f in div_columns:
        if f == feature:
            continue
        upsampled_frame['{0}_{1}'.format(feature, f)] = upsampled_frame[f] / upsampled_frame[feature]

features = []
[features.append(column) for column in upsampled_frame.columns if 'class' not in column]
print(features)

train_f, test_f = train_test_split(upsampled_frame_std, test_size = 0.2, 
                                   random_state = 72)

# --------------------------
# TPOT Generated Model
from sklearn.ensemble import ExtraTreesClassifier, VotingClassifier
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import FunctionTransformer
from sklearn.tree import DecisionTreeClassifier

exported_pipeline = make_pipeline(
    make_union(VotingClassifier([("est", ExtraTreesClassifier(criterion="entropy", max_features=0.36, n_estimators=500))]), FunctionTransformer(lambda X: X)),
    DecisionTreeClassifier()
)

exported_pipeline.fit(train_f[features], train_f['class'])

exported_pipeline.score(test_f[features], test_f['class'])

result = exported_pipeline.predict(test_f[features])

from sklearn.metrics import confusion_matrix
from classification_utilities import display_cm, display_adj_cm

conf = confusion_matrix(test_f['class'], result)
display_cm(conf, facies_labels, hide_zeros = True, display_metrics = True)

def accuracy(conf):
    total_correct = 0
    nb_classes = conf.shape[0]
    for i in np.arange(0, nb_classes):
        total_correct += conf[i][i]
    acc = total_correct / sum(sum(conf))
    return acc

print (accuracy(conf))

adjacent_facies = np.array([[1], [0, 2], [1], [4], [3, 5], [4, 6, 7], [5, 7], [5, 6, 8], [6, 7]])

def accuracy_adjacent(conf, adjacent_facies):
    nb_classes = conf.shape[0]
    total_correct = 0
    for i in np.arange(0, nb_classes):
        total_correct += conf[i][i]
        for j in adjacent_facies[i]:
            total_correct += conf[i][j]
    return total_correct / sum(sum(conf))

print(accuracy_adjacent(conf, adjacent_facies))            

test_path = '../validation_data_nofacies.csv'

# Read training data to dataframe
test = pd.read_csv(test_path)

# Set string features to integers

for i, value in enumerate(test['Formation'].unique()):
    test.loc[test['Formation'] == value, 'Formation'] = i
    
for i, value in enumerate(test['Well Name'].unique()):
    test.loc[test['Well Name'] == value, 'Well Name'] = i

# The first thing that will be done is to upsample and interpolate the training data,
# the objective here is to provide significantly more samples to train the regressor on and
# also to capture more of the sample interdependancy.
upsampled_arrays = []
test['orig_index'] = test.index

# Use rolling windows through upsampled frame, grouping by well name.

# Empty list to hold frames
mean_frames = []
above = []
below = []

for well, group in test.groupby('Well Name'):
    # Empty list to hold rolling frames
    constructor_list = []
    for f in resample_factors:
        
        working_frame = group[['Depth', 'GR', 'ILD_log10', 'DeltaPHI', 'PHIND', 'PE', 'NM_M',
       'RELPOS']]
        
        mean_frame = working_frame.rolling(window = f, center = True).mean().interpolate(method = 'index', limit_direction = 'both', limit = None)
        mean_frame.columns = ['Mean_{0}_{1}'.format(f, column) for column in mean_frame.columns]
        max_frame = working_frame.rolling(window = f, center = True).max().interpolate(method = 'index', limit_direction = 'both', limit = None)
        max_frame.columns = ['Max_{0}_{1}'.format(f, column) for column in max_frame.columns]
        min_frame = working_frame.rolling(window = f, center = True).min().interpolate(method = 'index', limit_direction = 'both', limit = None)
        min_frame.columns = ['Min_{0}_{1}'.format(f, column) for column in min_frame.columns]
        std_frame = working_frame.rolling(window = f, center = True).std().interpolate(method = 'index', limit_direction = 'both', limit = None)
        std_frame.columns = ['Std_{0}_{1}'.format(f, column) for column in std_frame.columns]
        var_frame = working_frame.rolling(window = f, center = True).var().interpolate(method = 'index', limit_direction = 'both', limit = None)
        var_frame.columns = ['Var_{0}_{1}'.format(f, column) for column in var_frame.columns]
        diff_frame = working_frame.diff(f, axis = 0).interpolate(method = 'index', limit_direction = 'both', limit = None)
        diff_frame.columns = ['Diff_{0}_{1}'.format(f, column) for column in diff_frame.columns]
        rdiff_frame = working_frame.sort_index(ascending = False).diff(f, axis = 0).interpolate(method = 'index', limit_direction = 'both', limit = None).sort_index()
        rdiff_frame.columns = ['Rdiff_{0}_{1}'.format(f, column) for column in rdiff_frame.columns]
        skew_frame = working_frame.rolling(window = f, center = True).skew().interpolate(method = 'index', limit_direction = 'both', limit = None)
        skew_frame.columns = ['Skew_{0}_{1}'.format(f, column) for column in skew_frame.columns]
        
        f_frame = pd.concat((mean_frame, max_frame, min_frame, std_frame, var_frame, diff_frame, rdiff_frame), axis = 1)
        
        constructor_list.append(f_frame)
        
    well_frame = pd.concat(constructor_list, axis = 1)
    well_frame['Well Name'] = well
    # orig index is holding the original index locations, to make extracting the results trivial
    well_frame['orig_index'] = group['orig_index']
    df = group.sort_values('Depth')
    u = df.shift(-1).fillna(method = 'ffill')
    b = df.shift(1).fillna(method = 'bfill')
    above.append(u[div_columns])
    below.append(b[div_columns])
    
    mean_frames.append(well_frame.fillna(method = 'bfill').fillna(method = 'ffill'))
    
frame = test
frame.index = frame['orig_index']
frame.drop(['orig_index', 'Well Name'], axis = 1, inplace = True)

for f in mean_frames:
    f.index = f['orig_index']

rolling_frame = pd.concat(mean_frames, axis = 0)
above_frame = pd.concat(above)
above_frame.columns = ['above_'+ column for column in above_frame.columns]
below_frame = pd.concat(below)
below_frame.columns = ['below_'+ column for column in below_frame.columns]
upsampled_frame = pd.concat((frame, rolling_frame, above_frame, below_frame), axis = 1)

features = [feature for feature in upsampled_frame.columns if 'class' not in feature]

std_scaler = preprocessing.StandardScaler().fit(upsampled_frame[features])
train_std = std_scaler.transform(upsampled_frame[features])

train_std_frame = upsampled_frame
for i, column in enumerate(features):
    train_std_frame.loc[:, column] = train_std[:, i]

upsampled_frame_std = train_std_frame

for feature in div_columns:
    for f in div_columns:
        if f == feature:
            continue
        upsampled_frame['{0}_{1}'.format(feature, f)] = upsampled_frame[f] / upsampled_frame[feature]
        
features = [feature for feature in upsampled_frame.columns if 'class' not in feature]

# Predict result on full sample set
result = exported_pipeline.predict(upsampled_frame[features])
# Add result to test set
upsampled_frame['Facies'] = result
# Output to csv
upsampled_frame.to_csv('05 - Well Facies Prediction - Test Data Set.csv')



