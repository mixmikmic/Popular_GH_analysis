get_ipython().magic('pylab inline')
import GPy
import GPyOpt
import numpy as np
from sklearn import svm
from GPy.models import GPRegression
from numpy.random import seed
np.random.seed(12345)

# Let's load the dataset
GPy.util.datasets.authorize_download = lambda x: True
data = GPy.util.datasets.olympic_marathon_men()
X = data['X']
Y = data['Y']
X_train = X[:20]
Y_train = Y[:20]
X_test = X[20:]
Y_test = Y[20:]

k =  GPy.kern.Matern32(1, variance=2, lengthscale=1)   + GPy.kern.Linear(1, variances=1)   + GPy.kern.Bias(1, variance=5)

m = GPRegression(X_train, Y_train, kernel=k,
                 normalizer=True)
print m

Y_train_pred, Y_train_pred_var = m.predict(X_train)
Y_test_pred, Y_test_pred_var = m.predict(X_test)

plot(X_train,Y_train_pred,'b',label='pred-train')
plot(X_test,Y_test_pred,'g',label='pred-test')

plot(X_train,Y_train,'rx',label='ground truth')
plot(X_test,Y_test,'rx')
legend(loc='best')

# Model parameters
m.parameter_names()

# List containing the description of the domain where we will perform the optmisation
domain = [
{'name': 'Mat32.variance',          'type': 'continuous', 'domain': (1,4.)},
{'name': 'Mat32.lengthscale',       'type': 'continuous', 'domain': (50.,150.)},
{'name': 'Linear.variances',        'type': 'continuous', 'domain': (1e-5,6)},
{'name': 'Bias.variance',           'type': 'continuous', 'domain': (1e-5,6)},
{'name': 'Gaussian_noise.variance', 'type': 'continuous', 'domain': (1e-5,4.)}
]

def f_lik(x):
    m[:] = x
    return m.objective_function()

opt = GPyOpt.methods.BayesianOptimization(f = f_lik,                  
                                          domain = domain,
                                          normalize_Y= True,
                                          exact_feval = True,
                                          model_type= 'GP',
                                          acquisition_type ='EI',       
                                          acquisition_jitter = 0.25)   

# it may take a few seconds
opt.run_optimization(max_iter=100)
opt.plot_convergence()

x_best = opt.X[np.argmin(opt.Y)].copy()
m[:] = x_best
print("The best model optimized with GPyOpt:")
print m

Y_train_pred, Y_train_pred_var = m.predict(X_train)
Y_test_pred, Y_test_pred_var = m.predict(X_test)

plot(X_train,Y_train_pred,'b',label='pred-train')
plot(X_test,Y_test_pred,'g',label='pred-test')

plot(X_train,Y_train,'rx',label='ground truth')
plot(X_test,Y_test,'rx')
legend(loc='best')

m[:] = opt.X[0].copy()

m.kern.Mat32.variance.constrain_bounded(1,4.)
m.kern.Mat32.lengthscale.constrain_bounded(50,150.)
m.kern.linear.variances.constrain_bounded(1e-5,6)
m.kern.bias.variance.constrain_bounded(1e-5,6)
m.Gaussian_noise.variance.constrain_bounded(1e-5,4.)

m.kern

m.optimize()

Y_train_pred, Y_train_pred_var = m.predict(X_train)
Y_test_pred, Y_test_pred_var = m.predict(X_test)

plot(X_train,Y_train_pred,'b',label='pred-train')
plot(X_test,Y_test_pred,'g',label='pred-test')

plot(X_train,Y_train,'rx',label='ground truth')
plot(X_test,Y_test,'rx')
legend(loc='best')

print("The best model optimized with the default optimizer:")
print m

