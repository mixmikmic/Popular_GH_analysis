from sklearn.cluster import KMeans

get_ipython().magic('pinfo KMeans')

from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)

cluster = KMeans(n_clusters=3)

cluster.fit(X)

cluster.predict(X)

from sklearn.tree import DecisionTreeClassifier

m = DecisionTreeClassifier(max_depth=2)

m.fit(cluster.predict(X)[:, None], y)

m.score(cluster.predict(X)[:, None], y)

from sklearn.decomposition import PCA

get_ipython().magic('pinfo PCA')

from sklearn.svm import SVC

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X)

X_pca.shape

SVC().fit(X_pca, y).score(X_pca, y)



