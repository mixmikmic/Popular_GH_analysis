import numpy as np
import pandas as pd
 
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso, lars_path
from sklearn.preprocessing import PolynomialFeatures

import warnings; warnings.simplefilter('ignore')

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns

dataset_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
data = pd.read_csv(dataset_url, sep=';')
data.head()

sns.set(style="ticks")
sns.pairplot(data)
plt.show()



X = data.drop('quality', axis=1)
y = data.quality
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    stratify=y)

 

poly = PolynomialFeatures(degree=3)
X_train_p = poly.fit_transform(X_train)

# ridge parameter shrinkage
l_alpha = np.linspace(-5,5)
params=[]
for i in l_alpha:
    ridge=Ridge(alpha=np.exp(i), fit_intercept=False)
    ridge.fit(X_train_p, y_train)
    params.append(ridge.coef_)

fig,ax = plt.subplots()


ax.plot(np.exp(l_alpha), params)
ax.set_xscale('log')
ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis
plt.xlabel('alpha')
plt.ylabel('weights')
plt.title('Parameters as a function of the ridge regularization')
plt.axis('tight')
plt.show()

fig.savefig('results.png')



params=[]
for i in l_alpha:
    lasso=Lasso(alpha=np.exp(i), fit_intercept=False)
    lasso.fit(X_train_p, y_train)
    params.append(lasso.coef_)
    
ax = plt.gca()

ax.plot(np.exp(l_alpha), params)
ax.set_xscale('log')
ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis
plt.xlabel('alpha')
plt.ylabel('weights')
plt.title('Lasso coefficients as a function of the regularization')
plt.axis('tight')
plt.show()

alphas, _, coefs = lars_path(X_train_p, y_train, method='lasso', verbose=True)
xx = np.sum(np.abs(coefs.T), axis=1)
xx /= xx[-1]

plt.plot(xx, coefs.T)
ymin, ymax = plt.ylim()
plt.vlines(xx, ymin, ymax, linestyle='dashed')
plt.xlabel('|coef| / max|coef|')
plt.ylabel('Coefficients')
plt.title('LASSO Path')
plt.axis('tight')
plt.show()

img=mpimg.imread('lasso_vs_ridge_regression.png')
imgplot = plt.imshow(img)
plt.show()

ridgeCV=RidgeCV(alphas=np.exp(l_alpha), normalize=True,cv=10)
ridgeCV.fit(X_train_p, y_train)

X_test_p=poly.fit_transform(X_test)
ridgeCV.predict(X_test_p)
ridge_score=ridgeCV.score(X_test_p, y_test, sample_weight=None)

ridgeCV.coef_


lassoCV=LassoCV(alphas=np.exp(l_alpha), normalize=True,cv=10)
lassoCV.fit(X_train_p, y_train)

lassoCV.predict(X_test_p)
lasso_score=lassoCV.score(X_test_p, y_test, sample_weight=None)

lassoCV.coef_

ridge_score

lasso_score

