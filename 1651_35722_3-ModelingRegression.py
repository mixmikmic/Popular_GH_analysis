import h2o
import re
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')
sns.set()

h2o.init()

train = h2o.upload_file("train_FD004_processed.csv")
test  = h2o.upload_file("test_FD004_processed.csv")

# Setup the column names of the training file
index_columns_names =  ["UnitNumber","Cycle"]

weight_column = "Cycle"

# And the name of the to be engineered target variable
dependent_var = ['RemainingUsefulLife']

independent_vars = [column_name for column_name in train.columns if re.search("CountInMode|stdized_SensorMeasure", column_name)]
independent_vars

fold_column_name = "FoldColumn"
train[fold_column_name] = train["UnitNumber"] % 5

from h2o.estimators.gbm import H2OGradientBoostingEstimator

gbm_regressor = H2OGradientBoostingEstimator(distribution="laplace", 
                                             score_each_iteration=True,
                                             stopping_metric="MSE", 
                                             stopping_tolerance=0.001,
                                             stopping_rounds=5,
                                             max_depth=10, ntrees=300)
gbm_regressor.train(x=independent_vars, y=dependent_var, 
                    training_frame=train, weights_column=weight_column,
                    fold_column=fold_column_name)

gbm_regressor

best_model = gbm_regressor

def sensor_preds(frame):
    frame["predict"] = ((frame["predict"] < 0.).ifelse(0., frame["predict"]))[0]
    return frame

models_for_pred = [best_model]+best_model.xvals
preds = [ sensor_preds(model.predict(test)) for model in models_for_pred ]
index = test[["UnitNumber","Cycle"]]
for i,pred in enumerate(preds):
    if i == 0:  # special handling for first model
        predictions = index.cbind(preds[i])
    else:
        predictions = predictions.cbind(preds[i])

predictions_df = predictions.as_data_frame(use_pandas=True)

# state is represented as [RUL, -1]
n_dim_state=2
n_dim_obs=len(preds)
a_transition_matrix = np.array([[1,1],[0,1]]) # Dynamics take state of [RUL, -1] and transition to [RUL-1, -1]
r_observation_covariance = np.diag( [ model.mse() for model in models_for_pred ] )
h_observation_matrices = np.array([[1,0] for _ in models_for_pred])

import pykalman as pyk

final_ensembled_preds = {}
pred_cols = [ name for name in predictions_df.columns if "predict" in name]
for unit in predictions_df.UnitNumber.unique():
    preds_for_unit = predictions_df[ predictions_df.UnitNumber == unit ]
    observations = preds_for_unit.as_matrix(pred_cols)
    initial_state_mean = np.array( [np.mean(observations[0]),-1] )
    kf = pyk.KalmanFilter(transition_matrices=a_transition_matrix,                          initial_state_mean=initial_state_mean,                          observation_covariance=r_observation_covariance,                          observation_matrices=h_observation_matrices,                          n_dim_state=n_dim_state, n_dim_obs=n_dim_obs)
    mean,_ = kf.filter(observations)
    final_ensembled_preds[unit] = mean

final_preds = { k:final_ensembled_preds[k][-1][0] for k in final_ensembled_preds.keys() }

final_preds_df = pd.DataFrame.from_dict(final_preds,orient='index')
final_preds_df.columns = ['predicted']

sns.tsplot(predictions_df[ predictions_df.UnitNumber == 2 ]["predict"])

sns.tsplot(final_ensembled_preds[2].T[0])

actual_RUL = pd.read_csv("RUL_FD004.txt",header=None,names=["actual"])
actual_RUL.index = actual_RUL.index+1
actual_preds = actual_RUL.join(final_preds_df)

def score(x):
    diff = x.predicted-x.actual
    result = np.expm1(diff/-13.) if diff < 0. else np.expm1(diff/10.)
    return result

actual_preds["score"] = actual_preds.apply(score, axis=1)
sum(actual_preds.score)/len(actual_preds)

g = sns.regplot("actual", "predicted", data=actual_preds, fit_reg=False)
g.set(xlim=(0, 160), ylim=(0, 180));
g.axes.plot((0, 160), (0, 160), c=".2", ls="--");

h2o.shutdown(prompt=False)

