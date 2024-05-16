from sklearn.datasets import load_iris
holder = load_iris()
X, y = holder.data, holder.target

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state =0)

from sklearn.linear_model import LinearRegression
lr = LinearRegression().fit(X_train,y_train)
# here we call the predict methond
y_pred = lr.predict(X_test)

from sklearn.svm import SVC
svc = SVC(kernel='linear').fit(X, y)
# here we call the predict methond
y_pred = svc.predict(X_test)

from sklearn import neighbors
knn = neighbors.KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)
# here we call the predict methond
y_pred = knn.predict_proba(X_test)

from sklearn.cluster import KMeans
k_means = KMeans(n_clusters=3, random_state=0).fit(X_train, y_train)
# here we call the predict methond
y_pred = k_means.predict(X_test)



