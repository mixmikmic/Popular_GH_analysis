import pandas as pd
import bokeh.plotting as bk
import numpy as np

from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from tpot import TPOTClassifier, TPOTRegressor

import sys
sys.path.append(r'C:\Users\george.crowther\Documents\Python\Projects\2016-ml-contest-master')

import classification_utilities

from IPython.core.display import display, HTML
display(HTML("<style>.container { width:80% !important; }</style>"))

bk.output_notebook()

# Input file paths
train_path = r'..\training_data.csv'

# Read training data to dataframe
train = pd.read_csv(train_path)

# TPOT library requires that the target class is renamed to 'class'
train.rename(columns={'Facies': 'class'}, inplace=True)

well_names = train['Well Name']

# Set string features to integers

for i, value in enumerate(train['Formation'].unique()):
    train.loc[train['Formation'] == value, 'Formation'] = i
    
for i, value in enumerate(train['Well Name'].unique()):
    train.loc[train['Well Name'] == value, 'Well Name'] = i

# The first thing that will be done is to upsample and interpolate the training data,
# the objective here is to provide significantly more samples to train the regressor on and
# also to capture more of the sample interdependancy.
upsampled_arrays = []
train['orig_index'] = train.index

for well, group in train.groupby('Well Name'):
    # This is a definite, but helpful, mis-use of the pandas resample timeseries
    # functionality.
    group.index = pd.to_datetime(group['Depth'] * 10)
    # Upsampled by a factor of 5 and interpolate
    us_group = group.resample('1ns').mean().interpolate(how='time')
    # Revert to integer
    us_group.index = us_group.index.asi8 / 10
    us_group['Well Name'] = well
    
    upsampled_arrays.append(us_group)

upsampled_arrays[0].head()

resample_factors = [2, 5, 10, 50, 100, 200]

initial_columns = ['Formation', 'Well Name', 'Depth', 'GR', 'ILD_log10',
       'DeltaPHI', 'PHIND', 'PE', 'NM_M', 'RELPOS']

upsampled_frame = pd.concat(upsampled_arrays, axis = 0)

# Use rolling windows through upsampled frame, grouping by well name.

# Empty list to hold frames
mean_frames = []

for well, group in upsampled_frame.groupby('Well Name'):
    # Empty list to hold rolling frames
    constructor_list = []
    for f in resample_factors:
        
        working_frame = group[['Depth', 'GR', 'ILD_log10', 'DeltaPHI', 'PHIND', 'PE', 'NM_M',
       'RELPOS', 'Well Name']]
        
        mean_frame = working_frame.rolling(window = f, center = True).mean().interpolate(method = 'index', limit_direction = 'both', limit = f)
        mean_frame.columns = ['Mean_{0}_{1}'.format(f, column) for column in mean_frame.columns]
        max_frame = working_frame.rolling(window = f, center = True).max().interpolate(method = 'index', limit_direction = 'both', limit = f)
        max_frame.columns = ['Max_{0}_{1}'.format(f, column) for column in max_frame.columns]
        min_frame = working_frame.rolling(window = f, center = True).min().interpolate(method = 'index', limit_direction = 'both', limit = f)
        min_frame.columns = ['Min_{0}_{1}'.format(f, column) for column in min_frame.columns]
        std_frame = working_frame.rolling(window = f, center = True).std().interpolate(method = 'index', limit_direction = 'both', limit = f)
        std_frame.columns = ['Std_{0}_{1}'.format(f, column) for column in std_frame.columns]
        var_frame = working_frame.rolling(window = f, center = True).var().interpolate(method = 'index', limit_direction = 'both', limit = f)
        var_frame.columns = ['Var_{0}_{1}'.format(f, column) for column in var_frame.columns]
        diff_frame = working_frame.diff(f, axis = 0).interpolate(method = 'index', limit_direction = 'both', limit = f)
        diff_frame.columns = ['Diff_{0}_{1}'.format(f, column) for column in diff_frame.columns]
        rdiff_frame = working_frame.sort_index(ascending = False).diff(f, axis = 0).interpolate(method = 'index', limit_direction = 'both', limit = f).sort_index()
        rdiff_frame.columns = ['Rdiff_{0}_{1}'.format(f, column) for column in rdiff_frame.columns]
        
        f_frame = pd.concat((mean_frame, max_frame, min_frame, std_frame, var_frame, diff_frame, rdiff_frame), axis = 1)
        
        constructor_list.append(f_frame)
        
    well_frame = pd.concat(constructor_list, axis = 1)
    well_frame['class'] = group['class']
    well_frame['Well Name'] = group['Well Name']
    # orig index is holding the original index locations, to make extracting the results trivial
    well_frame['orig_index'] = group['orig_index']
    mean_frames.append(well_frame)

upsampled_frame.index = upsampled_frame['orig_index']
upsampled_frame.drop(['orig_index', 'class', 'Well Name'], axis = 1, inplace = True)

for f in mean_frames:
    f.index = f['orig_index']

rolling_frame = pd.concat(mean_frames, axis = 0)
upsampled_frame = pd.concat((upsampled_frame, rolling_frame), axis = 1)

# Features is the column set used for training the model
features = upsampled_frame.columns[:-4]
print(features)

# Define model

from sklearn.ensemble import ExtraTreesRegressor, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import FunctionTransformer

exported_pipeline = make_pipeline(
    ExtraTreesRegressor(max_features=0.27, n_estimators=500)
)

# Fit model to data
exported_pipeline.fit(upsampled_frame[features], upsampled_frame['class'])

test_path = r'..\validation_data_nofacies.csv'

# Read training data to dataframe
test = pd.read_csv(test_path)

# TPOT library requires that the target class is renamed to 'class'
test.rename(columns={'Facies': 'class'}, inplace=True)

# Set string features to integers

for i, value in enumerate(test['Formation'].unique()):
    test.loc[train['Formation'] == value, 'Formation'] = i
    
for i, value in enumerate(test['Well Name'].unique()):
    test.loc[test['Well Name'] == value, 'Well Name'] = i

# The first thing that will be done is to upsample and interpolate the training data,
# the objective here is to provide significantly more samples to train the regressor on and
# also to capture more of the sample interdependancy.
upsampled_arrays = []
test['orig_index'] = test.index

for well, group in test.groupby('Well Name'):
    # This is a definite, but helpful, mis-use of the pandas resample timeseries
    # functionality.
    group.index = pd.to_datetime(group['Depth'] * 10)
    # Upsampled by a factor of 5 and interpolate
    us_group = group.resample('1ns').mean().interpolate(how='time')
    # Revert to integer
    us_group.index = us_group.index.asi8 / 10
    us_group['Well Name'] = well
    
    upsampled_arrays.append(us_group)
    
upsampled_frame = pd.concat(upsampled_arrays, axis = 0)

# Use rolling windows through upsampled frame, grouping by well name.

# Empty list to hold frames
mean_frames = []

for well, group in upsampled_frame.groupby('Well Name'):
    # Empty list to hold rolling frames
    constructor_list = []
    for f in resample_factors:
        
        working_frame = group[['Depth', 'GR', 'ILD_log10', 'DeltaPHI', 'PHIND', 'PE', 'NM_M',
       'RELPOS', 'Well Name']]
        
        mean_frame = working_frame.rolling(window = f, center = True).mean().interpolate(method = 'index', limit_direction = 'both', limit = f)
        mean_frame.columns = ['Mean_{0}_{1}'.format(f, column) for column in mean_frame.columns]
        max_frame = working_frame.rolling(window = f, center = True).max().interpolate(method = 'index', limit_direction = 'both', limit = f)
        max_frame.columns = ['Max_{0}_{1}'.format(f, column) for column in max_frame.columns]
        min_frame = working_frame.rolling(window = f, center = True).min().interpolate(method = 'index', limit_direction = 'both', limit = f)
        min_frame.columns = ['Min_{0}_{1}'.format(f, column) for column in min_frame.columns]
        std_frame = working_frame.rolling(window = f, center = True).std().interpolate(method = 'index', limit_direction = 'both', limit = f)
        std_frame.columns = ['Std_{0}_{1}'.format(f, column) for column in std_frame.columns]
        var_frame = working_frame.rolling(window = f, center = True).var().interpolate(method = 'index', limit_direction = 'both', limit = f)
        var_frame.columns = ['Var_{0}_{1}'.format(f, column) for column in var_frame.columns]
        diff_frame = working_frame.diff(f, axis = 0).interpolate(method = 'index', limit_direction = 'both', limit = f)
        diff_frame.columns = ['Diff_{0}_{1}'.format(f, column) for column in diff_frame.columns]
        rdiff_frame = working_frame.sort_index(ascending = False).diff(f, axis = 0).interpolate(method = 'index', limit_direction = 'both', limit = f).sort_index()
        rdiff_frame.columns = ['Rdiff_{0}_{1}'.format(f, column) for column in rdiff_frame.columns]
        
        f_frame = pd.concat((mean_frame, max_frame, min_frame, std_frame, var_frame, diff_frame, rdiff_frame), axis = 1)
        
        constructor_list.append(f_frame)
        
    well_frame = pd.concat(constructor_list, axis = 1)
    well_frame['Well Name'] = group['Well Name']
    # orig index is holding the original index locations, to make extracting the results trivial
    well_frame['orig_index'] = group['orig_index']
    mean_frames.append(well_frame)
    
upsampled_frame.index = upsampled_frame['orig_index']
upsampled_frame.drop(['orig_index', 'Well Name'], axis = 1, inplace = True)

for f in mean_frames:
    f.index = f['orig_index']

rolling_frame = pd.concat(mean_frames, axis = 0)
upsampled_frame = pd.concat((upsampled_frame, rolling_frame), axis = 1)

tfeatures = upsampled_frame.columns[:-3]
print(tfeatures)

# Predict result on full sample set
result = exported_pipeline.predict(upsampled_frame[tfeatures])
# Round result to nearest int
upsampled_frame['Facies'] = [round(n) for n in result]
# Extract results against test index
result_frame = upsampled_frame.loc[test.index, :]
# Output to csv
result_frame.to_csv('regressor_results.csv')



