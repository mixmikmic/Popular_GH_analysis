import h2o
h2o.connect()

import pandas as pd

titanic_df = pd.read_csv('/Users/avkashchauhan/learn/seattle-workshop/titanic_list.csv')

titanic_df.shape

titanic_df.count()

# Converting Pandas Frame to H2O Frame
titanic = h2o.H2OFrame(titanic_df)

titanic

# Note: You will see that the following command will not work
# Because it is a H2OFrame
titanic.count()

# The Other option to import data directly is to use H2O.
titanic_data = h2o.import_file('/Users/avkashchauhan/learn/seattle-workshop/titanic_list.csv')

titanic_data.shape

# Loading Estimators
from h2o.estimators.glm import H2OGeneralizedLinearEstimator

# set this to True if interactive (matplotlib) plots are desired
import matplotlib
interactive = False
if not interactive: matplotlib.use('Agg', warn=False)
import matplotlib.pyplot as plt

titanic_data.describe()

titanic_data.col_names

response = "survived"

# Selected Columns
# pclass, survived, sex, age, sibsp, parch, fare, embarked 
#

predictors = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]
# predictors = titanic_data.columns[:-1]

predictors

#Now setting factors to specific columns
titanic_data["pclass"] = titanic_data["pclass"].asfactor()
titanic_data["sex"] = titanic_data["sex"].asfactor()
titanic_data["embarked"] = titanic_data["embarked"].asfactor()

titanic_data.describe()

# Spliting the data set for training and validation
titanic_train, titanic_valid = titanic_data.split_frame(ratios=[0.9])

print(titanic_train.shape)
print(titanic_valid.shape)

titanic_glm = H2OGeneralizedLinearEstimator(alpha = .25)

titanic_glm.train(x = predictors, y = response, training_frame = titanic_train, validation_frame = titanic_valid)

# print the mse for the validation data
print "mse: ", titanic_glm.mse(valid=True)
print "r2 : ", titanic_glm.r2(valid=True)
print "rmse:", titanic_glm.rmse(valid=True)

# Note: Look for titanic_glm.[TAB] for the values you are interested into

# grid over `alpha`
# import Grid Search
from h2o.grid.grid_search import H2OGridSearch

# select the values for `alpha` to grid over
hyper_params = {'alpha': [0, .25, .5, .75, .1]}

# this example uses cartesian grid search because the search space is small
# and we want to see the performance of all models. For a larger search space use
# random grid search instead: {'strategy': "RandomDiscrete"}
# initialize the GLM estimator
titanic_glm_hype = H2OGeneralizedLinearEstimator()

# build grid search with previously made GLM and hyperparameters
titanitc_grid = H2OGridSearch(model = titanic_glm_hype, hyper_params = hyper_params,
                     search_criteria = {'strategy': "Cartesian"})

# train using the grid
titanitc_grid.train(x = predictors, y = response, training_frame = titanic_train, validation_frame = titanic_valid)

# sort the grid models by mse
titanic_sorted_grid = titanitc_grid.get_grid(sort_by='mse', decreasing=False)
print(titanic_sorted_grid)



# If you want to sort by r2 then try this
titanic_sorted_grid = titanitc_grid.get_grid(sort_by='r2', decreasing=False)
print(titanic_sorted_grid)

# Now adding alpha and lambda together
hyper_params = {'alpha': [0, .25, .5, .75, .1], 'lambda': [0, .1, .01, .001, .0001]}

titanic_glm_hype = H2OGeneralizedLinearEstimator()
titanitc_grid = H2OGridSearch(model = titanic_glm_hype, hyper_params = hyper_params,
                     search_criteria = {'strategy': "Cartesian"})
titanitc_grid.train(x = predictors, y = response, training_frame = titanic_train, validation_frame = titanic_valid)

# If you want to sort by r2 then try this
titanic_sorted_grid = titanitc_grid.get_grid(sort_by='r2', decreasing=False)
print(titanic_sorted_grid)



