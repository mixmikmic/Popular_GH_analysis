import numpy as np
import pandas as pd
import featuretools as ft
import utils

utils.download_data()
data_path = 'data/train_FD004.txt'
data = utils.load_data(data_path)

data.head()

from tqdm import tqdm

splits = 5
cutoff_time_list = []

for i in tqdm(range(splits)):
    cutoff_time_list.append(utils.make_cutoff_times(data))

cutoff_time_list[0].head()

from sklearn.cluster import KMeans

nclusters = 50

def make_entityset(data, nclusters, kmeans=None):
    X = data[['operational_setting_1', 'operational_setting_2', 'operational_setting_3']]
    if kmeans:
        kmeans=kmeans
    else:
        kmeans = KMeans(n_clusters=nclusters).fit(X)
    data['settings_clusters'] = kmeans.predict(X)
    
    es = ft.EntitySet('Dataset')
    es.entity_from_dataframe(dataframe=data,
                             entity_id='recordings',
                             index='index',
                             time_index='time')

    es.normalize_entity(base_entity_id='recordings', 
                        new_entity_id='engines',
                        index='engine_no')
    
    es.normalize_entity(base_entity_id='recordings', 
                        new_entity_id='settings_clusters',
                        index='settings_clusters')
    
    return es, kmeans
es, kmeans = make_entityset(data, nclusters)
es

from featuretools.primitives import Last, Max
from featuretools.primitives import make_agg_primitive
import featuretools.variable_types as vtypes

from tsfresh.feature_extraction.feature_calculators import (number_peaks, mean_abs_change, 
                                                            cid_ce, last_location_of_maximum, length)


Complexity = make_agg_primitive(lambda x: cid_ce(x, False),
                              input_types=[vtypes.Numeric],
                              return_type=vtypes.Numeric,
                              name="complexity")

fm, features = ft.dfs(entityset=es, 
                      target_entity='engines',
                      agg_primitives=[Last, Max, Complexity],
                      trans_primitives=[],
                      chunk_size=.26,
                      cutoff_time=cutoff_time_list[0],
                      max_depth=3,
                      verbose=True)

fm.to_csv('advanced_fm.csv')
fm.head()

fm_list = [fm]
for i in tqdm(range(1, splits)):
    fm = ft.calculate_feature_matrix(entityset=make_entityset(data, nclusters, kmeans=kmeans)[0], 
                                     features=features, 
                                     chunk_size=.26, 
                                     cutoff_time=cutoff_time_list[i])
    fm_list.append(fm)

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.feature_selection import RFE
def pipeline_for_test(fm_list, hyperparams=[100, 50, 50], do_selection=False):
    scores = []
    regs = []
    selectors = []
    for fm in fm_list:
        X = fm.copy().fillna(0)
        y = X.pop('RUL')
        reg = RandomForestRegressor(n_estimators=int(hyperparams[0]), 
                                    max_features=min(int(hyperparams[1]), 
                                                     int(hyperparams[2])))
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        if do_selection:
            reg2 = RandomForestRegressor(n_jobs=3)
            selector = RFE(reg2, int(hyperparams[2]), step=25)
            selector.fit(X_train, y_train)
            X_train = selector.transform(X_train)
            X_test = selector.transform(X_test)
            selectors.append(selector)
        reg.fit(X_train, y_train)
        regs.append(reg)
        
        preds = reg.predict(X_test)
        scores.append(mean_absolute_error(preds, y_test))
    return scores, regs, selectors    
scores, regs, selectors = pipeline_for_test(fm_list)
print([float('{:.1f}'.format(score)) for score in scores])
print('Average MAE: {:.1f}, Std: {:.2f}\n'.format(np.mean(scores), np.std(scores)))

most_imp_feats = utils.feature_importances(fm_list[0], regs[0])

data_test = utils.load_data('data/test_FD004.txt')

es_test, _ = make_entityset(data_test, nclusters, kmeans=kmeans)
fm_test = ft.calculate_feature_matrix(entityset=es_test, features=features, verbose=True, chunk_size='cutoff time')
X = fm_test.copy().fillna(0)
y = pd.read_csv('data/RUL_FD004.txt', sep=' ', header=-1, names=['RUL'], index_col=False)
preds = regs[0].predict(X)
print('Mean Abs Error: {:.2f}'.format(mean_absolute_error(preds, y)))

from btb.hyper_parameter import HyperParameter
from btb.tuning import GP

def run_btb(fm_list, n=30):
    hyperparam_ranges = [
            ('n_estimators', HyperParameter('int', [10, 200])),
            ('max_feats', HyperParameter('int', [5, 50])),
            ('nfeats', HyperParameter('int', [10, 70])),
    ]
    tuner = GP(hyperparam_ranges)

    tested_parameters = np.zeros((n, len(hyperparam_ranges)), dtype=object)
    scores = []
    
    print('[n_est, max_feats, nfeats]')
    best = 45

    for i in tqdm(range(n)):
        tuner.fit(tested_parameters[:i, :], scores)
        hyperparams = tuner.propose()
        cvscores, regs, selectors = pipeline_for_test(fm_list, hyperparams=hyperparams, do_selection=True)
        bound = np.mean(cvscores)
        tested_parameters[i, :] = hyperparams
        scores.append(-np.mean(cvscores))
        if np.mean(cvscores) + np.std(cvscores) < best:
            best = np.mean(cvscores)
            best_hyperparams = [name for name in hyperparams]
            best_reg = regs[0]
            best_sel = selectors[0]
            print('{}. {} -- Average MAE: {:.1f}, Std: {:.2f}'.format(i, 
                                                                      best_hyperparams, 
                                                                      np.mean(cvscores), 
                                                                      np.std(cvscores)))
            print('Raw: {}'.format([float('{:.1f}'.format(s)) for s in cvscores]))

    return best_hyperparams, (best_sel, best_reg)

best_hyperparams, best_pipeline = run_btb(fm_list, n=30)

X = fm_test.copy().fillna(0)
y = pd.read_csv('data/RUL_FD004.txt', sep=' ', header=-1, names=['RUL'], index_col=False)

preds = best_pipeline[1].predict(best_pipeline[0].transform(X))
score = mean_absolute_error(preds, y)
print('Mean Abs Error on Test: {:.2f}'.format(score))
most_imp_feats = utils.feature_importances(X.iloc[:, best_pipeline[0].support_], best_pipeline[1])

from featuretools.primitives import Min
old_fm, features = ft.dfs(entityset=es, 
                      target_entity='engines',
                      agg_primitives=[Last, Max, Min],
                      trans_primitives=[],
                      cutoff_time=cutoff_time_list[0],
                      max_depth=3,
                      verbose=True)

old_fm_list = [old_fm]
for i in tqdm(range(1, splits)):
    old_fm = ft.calculate_feature_matrix(entityset=make_entityset(data, nclusters, kmeans=kmeans)[0], 
                                     features=features, 
                                     cutoff_time=cutoff_time_list[i])
    old_fm_list.append(fm)

old_scores = []
median_scores = []
for fm in old_fm_list:
    X = fm.copy().fillna(0)
    y = X.pop('RUL')
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    reg = RandomForestRegressor()
    reg.fit(X_train, y_train)
    preds = reg.predict(X_test)
    old_scores.append(mean_absolute_error(preds, y_test))
    
    medianpredict = [np.median(y_train) for _ in y_test]
    median_scores.append(mean_absolute_error(medianpredict, y_test))

print([float('{:.1f}'.format(score)) for score in old_scores])
print('Average MAE: {:.2f}, Std: {:.2f}\n'.format(np.mean(old_scores), np.std(old_scores)))

print([float('{:.1f}'.format(score)) for score in median_scores])
print('Baseline by Median MAE: {:.2f}, Std: {:.2f}\n'.format(np.mean(median_scores), np.std(median_scores)))

y = pd.read_csv('data/RUL_FD004.txt', sep=' ', header=-1, names=['RUL'], index_col=False)
median_scores_2 = []
for ct in cutoff_time_list:
    medianpredict2 = [np.median(ct['RUL'].values) for _ in y.values]
    median_scores_2.append(mean_absolute_error(medianpredict2, y))
print([float('{:.1f}'.format(score)) for score in median_scores_2])
print('Baseline by Median MAE: {:.2f}, Std: {:.2f}\n'.format(np.mean(median_scores_2), np.std(median_scores_2)))



