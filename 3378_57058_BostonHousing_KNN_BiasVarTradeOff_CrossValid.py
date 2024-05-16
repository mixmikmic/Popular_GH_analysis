# enable In-Line MatPlotLib
get_ipython().magic('matplotlib inline')

# import:
from ggplot import aes, geom_line, geom_point, ggplot, ggtitle, xlab, ylab
from numpy import log, nan, sqrt
from os import system
from pandas import DataFrame, melt, read_csv
from random import seed
from sklearn.cross_validation import cross_val_score, KFold
from sklearn.neighbors import KNeighborsRegressor

system('pip install --upgrade git+git://GitHub.com/ChicagoBoothML/Helpy --no-dependencies')
from ChicagoBoothML_Helpy.CostFunctions import rmse

seed(99)

# read Boston Housing data into data frame
boston_housing = read_csv(
    'https://raw.githubusercontent.com/ChicagoBoothML/DATA___BostonHousing/master/BostonHousing.csv')
boston_housing.sort(columns='lstat', inplace=True)
nb_samples = len(boston_housing)
boston_housing

def plot_boston_housing_data(boston_housing_data,
                             x_name='lstat', y_name='medv', y_hat_name='predicted_medv',
                             title='Boston Housing: medv vs. lstat',
                             plot_predicted=True):
    g = ggplot(aes(x=x_name, y=y_name), data=boston_housing_data) +        geom_point(size=10, color='blue') +        ggtitle(title) +        xlab(x_name) + ylab(y_name)
    if plot_predicted:
        g += geom_line(aes(x=x_name, y=y_hat_name), size=2, color='darkorange')
    return g

plot_boston_housing_data(boston_housing, plot_predicted=False)

k = 5
knn_model = KNeighborsRegressor(n_neighbors=k)
knn_model.fit(X=boston_housing[['lstat']], y=boston_housing.medv)
boston_housing['predicted_medv'] = knn_model.predict(boston_housing[['lstat']])

plot_boston_housing_data(boston_housing, title='KNN Model with k = %i' %k)

k = 200
knn_model = KNeighborsRegressor(n_neighbors=k)
knn_model.fit(X=boston_housing[['lstat']], y=boston_housing.medv)
boston_housing['predicted_medv'] = knn_model.predict(boston_housing[['lstat']])

plot_boston_housing_data(boston_housing, title='KNN Model with k = %i' % k)

k = 50
knn_model = KNeighborsRegressor(n_neighbors=k)
knn_model.fit(X=boston_housing[['lstat']], y=boston_housing.medv)
boston_housing['predicted_medv'] = knn_model.predict(boston_housing[['lstat']])

plot_boston_housing_data(boston_housing, title='KNN Model with k = %i' % k)

# define Root-Mean-Square-Error scoring/evaluation function
# compliant with what SciKit Learn expects in this guide:
# http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.cross_val_score.html#sklearn.cross_validation.cross_val_score
def rmse_score(estimator, X, y):
    y_hat = estimator.predict(X)
    return rmse(y_hat, y)

NB_CROSS_VALIDATION_FOLDS = 5
NB_CROSS_VALIDATIONS = 6

k_range = range(2, 201)
cross_validations_avg_rmse_dataframe = DataFrame(dict(k=k_range, model_complexity=-log(k_range)))
cross_validations_avg_rmse_dataframe['cv_avg_rmse'] = 0.
cv_column_names = []
for v in range(NB_CROSS_VALIDATIONS):
    cv_column_name = 'cv_%i_rmse' % v
    cv_column_names.append(cv_column_name)
    cross_validations_avg_rmse_dataframe[cv_column_name] = nan
    for k in k_range:
        knn_model = KNeighborsRegressor(n_neighbors=k)
        avg_rmse_score = cross_val_score(
            knn_model,
            X=boston_housing[['lstat']],
            y=boston_housing.medv,
            cv=KFold(n=nb_samples,
                     n_folds=NB_CROSS_VALIDATION_FOLDS,
                     shuffle=True),
            scoring=rmse_score).mean()
        cross_validations_avg_rmse_dataframe.ix[
            cross_validations_avg_rmse_dataframe.k==k, cv_column_name] = avg_rmse_score
        
    cross_validations_avg_rmse_dataframe.cv_avg_rmse +=        (cross_validations_avg_rmse_dataframe[cv_column_name] -
         cross_validations_avg_rmse_dataframe.cv_avg_rmse) / (v + 1)
        
cross_validations_avg_rmse_longdataframe = melt(
    cross_validations_avg_rmse_dataframe,
    id_vars=['model_complexity', 'cv_avg_rmse'], value_vars=cv_column_names)

ggplot(aes(x='model_complexity', y='value', color='variable'),
       data=cross_validations_avg_rmse_longdataframe) +\
    geom_line(size=1, linetype='dashed') +\
    geom_line(aes(x='model_complexity', y='cv_avg_rmse'),
              data=cross_validations_avg_rmse_longdataframe,
              size=2, color='black') +\
    ggtitle('Cross Validations') +\
    xlab('Model Complexity (-log K)') + ylab('OOS RMSE')

best_k_index = cross_validations_avg_rmse_dataframe.cv_avg_rmse.argmin()
best_k = k_range[best_k_index]
best_k

k = best_k
knn_model = KNeighborsRegressor(n_neighbors=k)
knn_model.fit(X=boston_housing[['lstat']], y=boston_housing.medv)
boston_housing['predicted_medv'] = knn_model.predict(boston_housing[['lstat']])

plot_boston_housing_data(boston_housing, title='KNN Model with k = %i' % k)

