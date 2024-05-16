import featuretools as ft
from dask import bag
from dask.diagnostics import ProgressBar
import pandas as pd
import utils
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import os
ft.__version__

es = utils.load_entityset("partitioned_data/part_1/")
es

label_times = utils.make_labels(es=es,
                                product_name = "Banana",
                                cutoff_time = pd.Timestamp('March 15, 2015'),
                                prediction_window = ft.Timedelta("4 weeks"),
                                training_window = ft.Timedelta("60 days"))
label_times.head(5)

label_times["label"].value_counts()

feature_matrix, features = ft.dfs(target_entity="users", 
                                  cutoff_time=label_times,
                                  training_window=ft.Timedelta("60 days"), # same as above
                                  entityset=es,
                                  verbose=True)
# encode categorical values
fm_encoded, features_encoded = ft.encode_features(feature_matrix,
                                                  features)

print "Number of features %s" % len(features_encoded)
fm_encoded.head(10)

X = utils.merge_features_labels(fm_encoded, label_times)
X.drop(["user_id", "time"], axis=1, inplace=True)
X = X.fillna(0)
y = X.pop("label")

clf = RandomForestClassifier(n_estimators=400, n_jobs=-1)
scores = cross_val_score(estimator=clf,X=X, y=y, cv=3,
                         scoring="roc_auc", verbose=True)

"AUC %.2f +/- %.2f" % (scores.mean(), scores.std())

clf.fit(X, y)
top_features = utils.feature_importances(clf, features_encoded, n=20)

ft.save_features(top_features, "top_features")

pbar = ProgressBar()
pbar.register()

path = "partitioned_data/"
_, dirnames, _ = os.walk(path).next()
dirnames = [path+d for d in dirnames]
b = bag.from_sequence(dirnames)
entity_sets = b.map(utils.load_entityset)

label_times = entity_sets.map(utils.dask_make_labels,
                              product_name = "Banana",
                              cutoff_time = pd.Timestamp('March 1, 2015'),
                              prediction_window = ft.Timedelta("4 weeks"),
                              training_window = ft.Timedelta("60 days"))
label_times

# load in the features from before
top_features = ft.load_features("top_features", es)
feature_matrices = label_times.map(utils.calculate_feature_matrix, features=top_features)

fms_out = feature_matrices.compute()
X = pd.concat(fms_out)

X.drop(["user_id", "time"], axis=1, inplace=True)
y = X.pop("label")

clf = RandomForestClassifier(n_estimators=400, n_jobs=-1)
scores = cross_val_score(estimator=clf,X=X, y=y, cv=3,
                         scoring="roc_auc", verbose=True)

"AUC %.2f +/- %.2f" % (scores.mean(), scores.std())

clf.fit(X, y)
top_features = utils.feature_importances(clf, top_features, n=20)

