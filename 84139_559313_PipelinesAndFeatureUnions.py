from sklearn.pipeline import Pipeline

get_ipython().magic('pinfo Pipeline')

from sklearn.svm import SVC
from sklearn.decomposition import PCA
estimators = [('reduce_dim', PCA(n_components=2)), ('clf', SVC())]
pipe = Pipeline(estimators)
pipe 

from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)

# Notice no need to PCA the Xs in the score!
pipe.fit(X, y).score(X, y)

from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import Binarizer
make_pipeline(Binarizer(), MultinomialNB()) 

pipe.steps[0]

pipe.named_steps['reduce_dim']

pipe.set_params(clf__C=10) 

from sklearn.model_selection import GridSearchCV
params = dict(reduce_dim__n_components=[2, 5, 10],
              clf__C=[0.1, 10, 100])
grid_search = GridSearchCV(pipe, param_grid=params)

from sklearn.linear_model import LogisticRegression
params = dict(reduce_dim=[None, PCA(5), PCA(10)],
              clf=[SVC(), LogisticRegression()],
              clf__C=[0.1, 10, 100])
grid_search = GridSearchCV(pipe, param_grid=params)

from sklearn.pipeline import FeatureUnion
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
estimators = [('linear_pca', PCA()), ('kernel_pca', KernelPCA())]
combined = FeatureUnion(estimators)
combined 

combined.fit_transform(X).shape

combined.set_params(kernel_pca=None) 

combined.fit_transform(X).shape



