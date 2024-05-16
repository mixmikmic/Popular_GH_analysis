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
from sgd_regressor import sgd_regressor

class ridge_regressor(sgd_regressor):
    
    def __init__(self, n_iter=100, alpha=0.01, verbose=False, return_steps=False, fit_intercept=True, 
                 dynamic=True, loss='ols', epsilon=0.1, lamb=1e-6, l1_perc = 0.5):
        """
        Ridge Regressor - This is a wrapper on the SGD class where the regularization is set
        to the L2 Norm. All other functionality is the same as the SGD class.
        ---
        KWargs:
        
        n_iter: number of epochs to run in while fitting to the data. Total number of steps
        will be n_iter*X.shape[0]. 
        
        alpha: The learning rate. Moderates the step size during the gradient descent algorithm.
        
        verbose: Whether to print out coefficient information during the epochs
        
        return_steps: If True, fit returns a list of the coefficients at each update step for diagnostics
        
        fit_intercept: If True, an extra coefficient is added with no associated feature to act as the
                       base prediction if all X are 0.
                       
        dynamic: If true, an annealing scedule is used to scale the learning rate. 
        
        lamb: Stands for lambda. Sets the strength of the regularization. Large lambda causes large
              regression. If regularization is off, this does not apply to anything.
              
        l1_perc: If using elastic net, this variable sets what portion of the penalty is L1 vs L2. 
                 If regularize='EN' and l1_perc = 1, equivalent to regularize='L1'. If 
                 regularize='EN' and l1_perc = 0, equivalent to regulzarize='L2'.
        """
        self.coef_ = None
        self.trained = False
        self.n_iter = n_iter
        self.alpha_ = alpha
        self.verbosity = verbose
        self._return_steps = return_steps
        self._fit_intercept = fit_intercept
        self._next_alpha_shift = 0.1 # Only used if dynamic=True
        self._dynamic = dynamic
        self._regularize = 'L2'
        self._lamb = lamb
        self._l1_perc = l1_perc

def gen_data(rows = 200, gen_coefs = [2,4], gen_inter = 0):
    X = np.random.rand(rows,len(gen_coefs))
    y = np.sum(np.tile(np.array(gen_coefs),(X.shape[0],1))*X,axis=1)
    y = y + np.random.normal(0,0.5, size=X.shape[0])
    y = y + gen_inter
    return X, y

actual_coefs = [10,8,9,10,11]
X, y = gen_data(gen_coefs=actual_coefs[1:], gen_inter=actual_coefs[0])

import pandas as pd
cols = []
for i in range(X.shape[1]):
    cols.append('X'+str(i))
data = pd.DataFrame(X, columns=cols)
data['y'] = y
data.head()

ridge = ridge_regressor(n_iter=500, alpha=1e-3, verbose=False, dynamic=False, return_steps=True, lamb=1e-6)

steps = ridge.fit(data.iloc[:,:-1],data.iloc[:,-1])

ridge.coef_

test_X, test_y = gen_data(rows=200, gen_coefs=actual_coefs[1:], gen_inter=actual_coefs[0])
pred_y = sgd.predict(test_X)
test_err = pred_y - test_y

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('seaborn')

sns.distplot(test_err);

from scipy.stats import normaltest
print(normaltest(test_err))

plt.scatter(test_y, pred_y, s=50)
temp = np.linspace(min(test_y),max(test_y),100)
plt.plot(temp,temp,'r-')
plt.xlabel("True y")
plt.ylabel("Predicted y");

def plot_beta_space(steps, components = (0,1), last_300=False, zoom=False):
    plt.figure(figsize=(20,16))
    try:
        B0 = np.array(steps).T[components[0]]
        B1 = np.array(steps).T[components[1]]
    except:
        print("Couldn't find those components, defaulting to (0,1)")
        B0 = np.array(steps).T[0]
        B1 = np.array(steps).T[1]
    if last_300:
        steps_to_show=-300
        skip = 2
        plt.scatter(B0[steps_to_show::skip],B1[steps_to_show::skip],c=plt.cm.rainbow(np.linspace(0,1,len(B0[steps_to_show::skip]))));
        plt.scatter(steps[steps_to_show][0],steps[steps_to_show][1],c='r',marker='x', s=400,label='Start')
        plt.scatter(steps[-1][0],steps[-1][1],c='k',marker='x', s=400,label='End')
        plt.title("Movement in the Coefficient Space, Last "+str(-steps_to_show)+" steps!",fontsize=32);
    else: 
        plt.scatter(B0[::25],B1[::25],c=plt.cm.rainbow(np.linspace(0,1,len(B0[::25]))));
        plt.scatter(steps[0][0],steps[0][1],c='r',marker='x', s=400,label='Start')
        plt.scatter(steps[-1][0],steps[-1][1],c='k',marker='x', s=400,label='End')
        plt.title("Movement in the Coefficient Space",fontsize=32);
    plt.legend(fontsize=32, loc='upper left', frameon=True, facecolor='#FFFFFF', edgecolor='#333333');
    plt.xlabel("B"+str(components[0]),fontsize=26)
    plt.ylabel("B"+str(components[1]),fontsize=26);
    if zoom:
        plt.ylim(min(B1[steps_to_show::skip]), max(B1[steps_to_show::skip]))
        plt.xlim(min(B0[steps_to_show::skip]), max(B0[steps_to_show::skip]));

plot_beta_space(steps)

plot_beta_space(steps, last_300=True)

from mpl_toolkits.mplot3d import Axes3D
plt.style.use('seaborn')

def plot_beta_space3D(steps, components = (0,1)):
    def cost_function(x,y):
        return (x-actual_coefs[components[0]])**2 + (y-actual_coefs[components[1]])**2
    
    plot_vals_x = []
    plot_vals_y = []
    plot_vals_z = []
    for b1 in np.linspace(0,20,100):
        for b2 in np.linspace(0,20,100):
            cost = cost_function(b1,b2)
            plot_vals_x.append(b1)
            plot_vals_y.append(b2)
            plot_vals_z.append(cost)
    
    try:
        B0 = np.array(steps).T[components[0]]
        B1 = np.array(steps).T[components[1]]
    except:
        print("Couldn't find those components, defaulting to (0,1)")
        B0 = np.array(steps).T[0]
        B1 = np.array(steps).T[1]
    
    Z = cost_function(B0, B1)+10
    fig = plt.figure(figsize=(20,16))
    ax = fig.gca(projection='3d')
    ax.plot_trisurf(plot_vals_x,plot_vals_y,plot_vals_z, cmap=plt.cm.Blues, linewidth=0.2, alpha=0.4)
    ax.scatter(B0[::5],B1[::5],Z[::5],c='k',s=150);
    ax.set_xlabel("B0", fontsize=20, labelpad=20)
    ax.set_ylabel("B1", fontsize=20, labelpad=20)
    ax.set_zlabel("Cost Function Value", fontsize=20, labelpad=20);
    return ax

ax = plot_beta_space3D(steps)
ax.view_init(25, 75)

from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn')
get_ipython().run_line_magic('matplotlib', 'inline')

X = pd.DataFrame(StandardScaler().fit_transform(load_boston().data))
y = pd.DataFrame(load_boston().target)

X.describe()

# Simplest form - fit intercept, no dynamic learning for comparison
ridge = ridge_regressor(n_iter=1000, fit_intercept=True, lamb=1e-6)
ridge.fit(X.iloc[:100],y[:100])

ridge.coef_

pred = ridge.predict(X.iloc[100:])
plt.scatter(y.iloc[100:], pred, s=50, alpha=0.5)
temp = np.linspace(min(test_y),max(test_y),100)
plt.plot(temp,temp,'r-')
plt.xlabel("True y")
plt.ylabel("Predicted y");



