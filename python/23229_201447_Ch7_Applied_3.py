import pandas as pd
import numpy as np
import statsmodels.api as sm

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import cross_val_score
from patsy import cr, dmatrix
from pandas import scatter_matrix

from matplotlib import pyplot as plt
get_ipython().magic('matplotlib inline')
plt.style.use('ggplot')

np.set_printoptions(precision=4)

df = pd.read_csv('../../../data/Auto.csv', na_values='?')
df = df.dropna() # drop rows with na values
df.head()

pd.scatter_matrix(df, alpha=0.2, figsize=(12,8));

def natural_spline_cv(predictor, response, dofs=list(np.arange(2,10)), kfolds=5):
    """
    Returns an sklearn LinearRegression model object of a spline regression of predictor(pd.Series) onto response 
    (pd.Series). Uses kfold cross-validation and gan optionally return a plot .
    """
    # cross-val scores- array[dof]
    scores = np.array([])
    X_basis = np.array([])
    
    for dof in dofs:
        # natural spline dmatrix
        formula = r'1 + cr(predictor, df=%d, constraints="center")' %(dof)
        X_basis = dmatrix(formula, data={'predictor':predictor}, return_type='matrix')
     
        # model
        estimator = LinearRegression(fit_intercept=False)
        # cross-validation
        scores = np.append(scores, -np.mean(cross_val_score(estimator, X_basis, response, 
                                                            scoring='mean_squared_error', cv=kfolds)))
    # Build CV Error plot
    fig,ax = plt.subplots(1,1, figsize=(8,6))
    ax.plot(dofs, scores, lw=2, color='k', marker='o')
    ax.set_xlabel('Degrees of Freedom')
    ax.set_ylabel('CV Test MSE')
        

scores = natural_spline_cv(df.acceleration, df.mpg)

scores = natural_spline_cv(df.weight, df.mpg)



