import sklearn.datasets as datasets

X, y = datasets.load_boston(return_X_y=True)

print y[0]

from sklearn import linear_model

get_ipython().magic('pinfo linear_model.ElasticNet')

m = linear_model.ElasticNet(alpha=.1, l1_ratio=.9)

m.fit(X, y)

m.coef_

m.intercept_

m.predict([X[0]])

y[0]

m.score(X, y)

get_ipython().magic('pinfo m.score')

get_ipython().magic('pinfo linear_model.ElasticNetCV')

m = linear_model.ElasticNetCV(
    l1_ratio=[.1, .5, .7, .9, .95, .99, 1], 
    n_alphas=20)

m.fit(X, y)

m.alphas_

m.mse_path_

m.alpha_

m.l1_ratio_

m.predict([X[0]])

m.score(X, y)

X, y = datasets.load_iris(return_X_y=True)

d = datasets.load_iris()

print d.DESCR

get_ipython().magic('pinfo linear_model.LogisticRegressionCV')

m = linear_model.LogisticRegressionCV(Cs=10, n_jobs=2)

m.fit(X, y)

m.coef_

m.predict([X[0]])

y[0]

m.predict_proba([X[0]])

m.predict_log_proba([X[0]])

m.score(X, y)

get_ipython().magic('pinfo m.score')





