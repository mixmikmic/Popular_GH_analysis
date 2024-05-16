get_ipython().magic('matplotlib inline')
import numpy as np
from pycobra.cobra import cobra
from pycobra.visualisation import visualisation
from pycobra.diagnostics import diagnostics

# setting up our random data-set
rng = np.random.RandomState(42)

# D1 = train machines; D2 = create COBRA; D3 = calibrate epsilon, alpha; D4 = testing
n_features = 2
D1, D2, D3, D4 = 200, 200, 200, 200
D = D1 + D2 + D3 + D4
X = rng.uniform(-1, 1, D * n_features).reshape(D, n_features)
# Y = np.power(X[:,1], 2) + np.power(X[:,3], 3) + np.exp(X[:,10]) 
Y = np.power(X[:,0], 2) + np.power(X[:,1], 3)

# training data-set
X_train = X[:D1 + D2]
X_test = X[D1 + D2 + D3:D1 + D2 + D3 + D4]
X_eps = X[D1 + D2:D1 + D2 + D3]
# for testing
Y_train = Y[:D1 + D2]
Y_test = Y[D1 + D2 + D3:D1 + D2 + D3 + D4]
Y_eps = Y[D1 + D2:D1 + D2 + D3]

# set up our COBRA machine with the data
COBRA = cobra(X_train, Y_train, epsilon = 0.5)

cobra_vis = visualisation(COBRA, X_test, Y_test)

# to plot our machines, we need a linspace as input. This is the 'scale' to plot and should be the range of the results
# since our data ranges from -1 to 1 it is such - and we space it out to a hundred points
cobra_vis.plot_machines(machines=["COBRA"])

cobra_vis.plot_machines(y_test=Y_test)

cobra_vis.QQ(Y_test)

cobra_vis.boxplot()

indices, MSE = cobra_vis.indice_info(X_eps[0:50], Y_eps[0:50], epsilon=0.50)

cobra_vis.color_cobra(X_eps[0:50], indice_info=indices, single=True)

cobra_vis.color_cobra(X_eps[0:50], indice_info=indices)

cobra_vis.voronoi(X_eps[0:50], indice_info=indices, single=True)

cobra_vis.voronoi(X_eps[0:50], indice_info=indices)

cobra_vis.voronoi(X_eps[0:50], indice_info=indices, MSE=MSE, gradient=True)

