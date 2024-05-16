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

from scipy.stats import mode
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFwe, SelectKBest, f_classif, SelectPercentile
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, Binarizer, Normalizer, StandardScaler
from xgboost import XGBClassifier

models = [
    make_pipeline(
    MinMaxScaler(),
    XGBClassifier(learning_rate=0.02, max_depth=5, min_child_weight=20, n_estimators=500, subsample=0.19)
),
    make_pipeline(
    make_union(VotingClassifier([("est", LogisticRegression(C=0.13, dual=False, penalty="l1"))]), FunctionTransformer(lambda X: X)),
    RandomForestClassifier(n_estimators=500)
),
    make_pipeline(
    Binarizer(threshold=0.72),
    RandomForestClassifier(n_estimators=500)
),
    make_pipeline(
    make_union(VotingClassifier([("est", GradientBoostingClassifier(learning_rate=1.0, max_features=1.0, n_estimators=500))]), FunctionTransformer(lambda X: X)),
    BernoulliNB(alpha=28.0, binarize=0.85, fit_prior=True)
),
    make_pipeline(
    Normalizer(norm="l1"),
    make_union(VotingClassifier([("est", RandomForestClassifier(n_estimators=500))]), FunctionTransformer(lambda X: X)),
    SelectKBest(k=47, score_func=f_classif),
    SelectFwe(alpha=0.05, score_func=f_classif),
    RandomForestClassifier(n_estimators=500)
),
    make_pipeline(
    make_union(VotingClassifier([("est", LinearSVC(C=0.26, dual=False, penalty="l2"))]), FunctionTransformer(lambda X: X)),
    RandomForestClassifier(n_estimators=500)
),
    make_pipeline(
    Normalizer(norm="l2"),
    make_union(VotingClassifier([("est", ExtraTreesClassifier(criterion="entropy", max_features=0.3, n_estimators=500))]), FunctionTransformer(lambda X: X)),
    GaussianNB()
),
    make_pipeline(
    make_union(VotingClassifier([("est", BernoulliNB(alpha=49.0, binarize=0.06, fit_prior=True))]), FunctionTransformer(lambda X: X)),
    StandardScaler(),
    make_union(VotingClassifier([("est", GradientBoostingClassifier(learning_rate=0.87, max_features=0.87, n_estimators=500))]), FunctionTransformer(lambda X: X)),
    ExtraTreesClassifier(criterion="entropy", max_features=0.001, n_estimators=500)
),
    make_pipeline(
    make_union(VotingClassifier([("est", RandomForestClassifier(n_estimators=500))]), FunctionTransformer(lambda X: X)),
    BernoulliNB(alpha=1e-06, binarize=0.09, fit_prior=True)
),
    make_pipeline(
    Normalizer(norm="max"),
    MinMaxScaler(),
    RandomForestClassifier(n_estimators=500)
),
    make_pipeline(
    SelectPercentile(percentile=18, score_func=f_classif),
    RandomForestClassifier(n_estimators=500)
),
    make_pipeline(
    SelectKBest(k=50, score_func=f_classif),
    RandomForestClassifier(n_estimators=500)
),
    make_pipeline(
    XGBClassifier(learning_rate=0.51, max_depth=10, min_child_weight=20, n_estimators=500, subsample=1.0)
),
    make_pipeline(
    make_union(VotingClassifier([("est", KNeighborsClassifier(n_neighbors=5, weights="uniform"))]), FunctionTransformer(lambda X: X)),
    RandomForestClassifier(n_estimators=500)
),
    make_pipeline(
    StandardScaler(),
    SelectPercentile(percentile=19, score_func=f_classif),
    LinearSVC(C=0.02, dual=False, penalty="l1")
),
    make_pipeline(
    XGBClassifier(learning_rate=0.01, max_depth=10, min_child_weight=20, n_estimators=500, subsample=0.36)
)]

train_path = '../training_data.csv'
test_path = '../validation_data_nofacies.csv'

facies_labels = ['SS', 'CSiS', 'FSiS', 'SiSh', 'MS', 'WS', 'D', 'PS', 'BS']


def feature_extraction(file_path, retain_class = True):

    # Read training data to dataframe
    test = pd.read_csv(file_path)
    
    if 'Facies' in test.columns:
        test.rename(columns={'Facies': 'class'}, inplace=True)

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
 
    return upsampled_frame_std, features

train_data_set, features = feature_extraction(train_path)
test_data_set, test_features = feature_extraction(test_path)

train_data_set.head()

lpgo = LeavePGroupsOut(2)

split_list = []
fitted_models = []

for train, val in lpgo.split(train_data_set[features], 
                             train_data_set['class'], 
                             groups = train_data_set['Well Name']):
    hist_tr = np.histogram(train_data_set.loc[train, 'class'], 
                           bins = np.arange(len(facies_labels) + 1) + 0.5)
    hist_val = np.histogram(train_data_set.loc[val, 'class'],
                           bins = np.arange(len(facies_labels) + 1) + 0.5)
    if np.all(hist_tr[0] != 0) & np.all(hist_val[0] != 0):
        split_list.append({'train': train, 'val': val})
        
for s, split in enumerate(split_list):
    print('Split %d' % s)
    print('    training:   %s' % (train_data_set['Well Name'].loc[split['train']].unique()))
    print('    validation: %s' % (train_data_set['Well Name'].loc[split['val']].unique()))

fitted_models = []
r = []

for i, split in enumerate(split_list):

    # Select training and validation data from current split
    X_tr = train_data_set.loc[split['train'], features]
    X_v = train_data_set.loc[split['val'], features]
    y_tr = train_data_set.loc[split['train'], 'class']
    y_v = train_data_set.loc[split['val'], 'class']

    # Fit model from split
    fitted_models.append(models[i].fit(X_tr, y_tr))
    
    # Predict for model
    r.append(fitted_models[-1].predict(test_data_set[test_features]))
    
results = mode(np.vstack(r))[0][0]

test_data_set['Facies'] = results

test_data_set.iloc[:, ::-1].head()

test_data_set.iloc[:, ::-1].to_csv('06 - Combined Models.csv')



