import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
import math
import scipy
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('seaborn')

import numpy as np
import sklearn
import matplotlib
import pandas as pd
import sys
libraries = (('Matplotlib', matplotlib), ('Numpy', np), ('Pandas', pd))

print("Python Version:", sys.version, '\n')
for lib in libraries:
    print('{0} Version: {1}'.format(lib[0], lib[1].__version__))

import numpy as np
import pandas as pd
import sys
sys.path.append('../../modules')
from kmeans import kmeans

class spectral_clustering:
    
    def __init__(self, k=3, connectivity=20, svd_dims=3, affinity='neighbors', bandwidth=1.):
        self.k = k
        self.connect = connectivity
        self.dims = svd_dims
        if affinity in ['neighbors', 'rbf']:
            self.affinity_type = affinity
        else:
            print("Not a valid affinity type, default to 'neighbors'.")
            self.affinity_type = 'neighbors'
        self.bandwidth = bandwidth
    
    def rbf_kernel(self, x1, x2, sig=1.):
        """
        Returns the rbf affinity between two points (x1 and x2),
        for a given bandwidth (standard deviation).
        ---
        Inputs: 
            x1; point 1(array)
            x2; point 2(array)
            sig; standard deviation (float)
        """
        diff = np.sqrt(np.sum((x1-x2)**2))
        norm = 1/(np.sqrt(2*np.pi*sig**2))
        return norm*np.exp(-diff**2/(2*sig**2))
    
    def compute_distance_between_all_points(self, pt1, pts, connectivity=None):
        """
        Returns the distance between points. Currently only uses Euclidean distance.
        ---
        Input: data point, all data points (np arrays)
        Output: Distance (float)
        """
        if self.affinity_type == 'neighbors':
            x = np.sqrt(np.sum((pt1 - pts)**2, axis=1))
            idxs = x.argsort()[:connectivity]
            filt = np.ones(len(x), dtype=bool)
            filt[idxs] = False
            x[filt] = 0.
            x[~filt] = 1.
        elif self.affinity_type == 'rbf':
            x = []
            for p in pts:
                x.append(self.rbf_kernel(pt1, p, sig=self.bandwidth))
        return x
    
    def fit(self, X):
        X = self.pandas_to_numpy(X)
        self.original_data = np.copy(X)
        self.similarity = np.array([self.compute_distance_between_all_points(p,X, connectivity=self.connect) for p in X])
        self.similarity /= max(self.similarity.ravel())
        self.U, self.Sigma, self.VT = self.do_svd(self.similarity)
        self.kmeans = kmeans(k=self.k)
        self.kmeans.fit(self.U)
        
    def fit_predict(self, X):
        self.fit(X)
        return self.predict(X)
        
    def transform_to_svd_space(self,X):
        sig_inv = np.linalg.inv(self.Sigma)
        return np.dot(np.dot(X,self.U),sig_inv)
    
    def predict(self, X):
        X = self.pandas_to_numpy(X)
        sim_space = [self.compute_distance_between_all_points(p,self.original_data, connectivity=self.connect) for p in X]
        transformed_X = np.array([self.transform_to_svd_space(x) for x in sim_space])
        return self.kmeans.predict(transformed_X)
    
    def do_svd(self, similarity):
        dims = self.dims
        U, Sigma, VT = np.linalg.svd(similarity)
        VT = VT[:dims,:]
        U = U[:,:dims]
        Sigma = np.diag(Sigma[:dims])
        return U, Sigma, VT
        
    def plot_similarity_matrix(self):
        plt.figure(dpi=200)
        plt.imshow(self.similarity, cmap=plt.cm.Blues)
        plt.xlabel("Point ID", fontsize=16)
        plt.ylabel("Point ID", fontsize=16)
        plt.title("Similarity Matrix (1 for neighbors, 0 for not)", fontsize=16);
        plt.colorbar(cmap=plt.cm.Blues);
        
    def pandas_to_numpy(self, x):
        """
        Checks if the input is a Dataframe or series, converts to numpy matrix for
        calculation purposes.
        ---
        Input: X (array, dataframe, or series)
        
        Output: X (array)
        """
        if type(x) == type(pd.DataFrame()) or type(x) == type(pd.Series()):
            return x.as_matrix()
        if type(x) == type(np.array([1,2])):
            return x
        return np.array(x)

def get_data(n_clust = 3):
    X1 = np.random.normal(-5,1,50).reshape(-1,1)
    y1 = np.random.normal(-5,1,50).reshape(-1,1)
    for _ in range(n_clust-1):
        X2 = np.random.normal(np.random.randint(-10,10),1,50).reshape(-1,1)
        y2 = np.random.normal(np.random.randint(-10,10),1,50).reshape(-1,1)
        X1 = np.vstack((X1,X2)).reshape(-1,1)
        y1 = np.vstack((y1,y2)).reshape(-1,1)
    X = np.hstack((X1,y1))
    return X

X = get_data(n_clust=5)
#np.random.shuffle(X)
plt.scatter(X[:,0],X[:,1]);

sc = spectral_clustering(k=5)
preds = sc.fit_predict(X)

plt.scatter(X[:,0],X[:,1],c=preds);

for x, y in zip(X[:10],preds[:10]):
    print("Point at: ", x)
    print("Cluster Num: ", y)
print("---")
print("At [-5, -5], should be same cluster.")
print("Prediction is: ",sc.predict([[-5,-5]]))

sc.plot_similarity_matrix()

from sklearn.datasets import make_circles
X, y = make_circles(n_samples=400, factor=0.5, random_state=0, noise=0.05)
plt.scatter(X[:,0],X[:,1]);

sc = spectral_clustering(k=2, affinity='neighbors')
sc.fit(X)

sc.plot_similarity_matrix()

preds = sc.predict(X)
plt.scatter(X[:,0],X[:,1],c=preds);

sc = spectral_clustering(k=2, affinity='rbf', bandwidth=0.3, svd_dims=15)
sc.fit(X)

sc.plot_similarity_matrix()

preds = sc.predict(X)
plt.scatter(X[:,0],X[:,1],c=preds);



