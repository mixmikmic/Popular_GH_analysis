get_ipython().magic('pylab inline')
import GPy
import GPyOpt
from numpy.random import seed

def myf(x):
    return (2*x)**2

bounds = [{'name': 'var_1', 'type': 'continuous', 'domain': (-1,1)}]

max_iter = 15

myProblem = GPyOpt.methods.BayesianOptimization(myf,bounds)

myProblem.run_optimization(max_iter)

myProblem.x_opt

myProblem.fx_opt

from IPython.display import YouTubeVideo
YouTubeVideo('ualnbKfkc3Q')

get_ipython().magic('pylab inline')
import GPy
import GPyOpt

# Create the true and perturbed Forrester function and the boundaries of the problem
f_true= GPyOpt.objective_examples.experiments1d.forrester()          # noisy version
bounds = [{'name': 'var_1', 'type': 'continuous', 'domain': (0,1)}]  # problem constrains 

f_true.plot()

# Creates GPyOpt object with the model and anquisition fucntion
seed(123)
myBopt = GPyOpt.methods.BayesianOptimization(f=f_true.f,            # function to optimize       
                                             domain=bounds,        # box-constrains of the problem
                                             acquisition_type='EI',
                                             exact_feval = True) # Selects the Expected improvement

# Run the optimization
max_iter = 15     # evaluation budget
max_time = 60     # time budget 
eps      = 10e-6  # Minimum allows distance between the las two observations

myBopt.run_optimization(max_iter, max_time, eps)                     

myBopt.plot_acquisition()

myBopt.plot_convergence()

myBopt.plot_convergence()

# starts the optimization, 
import numpy as np
X_initial = np.array([[0.2],[0.4],[0.6]])

iterBopt = GPyOpt.methods.BayesianOptimization(f=f_true.f,                 
                                             domain=bounds,        
                                             acquisition_type='EI',
                                             X = X_initial,
                                             exact_feval = True,
                                             normalize_Y = False,
                                             acquisition_jitter = 0.01)

iterBopt.model.model.kern.variance.constrain_fixed(2.5)

iterBopt.plot_acquisition('./figures/iteration%.03i.png' % (0))

from IPython.display import clear_output
N_iter = 15

for i in range(N_iter):
    clear_output()
    iterBopt.run_optimization(max_iter=1) 
    iterBopt.plot_acquisition('./figures/iteration%.03i.png' % (i + 1))

# create the object function
f_true = GPyOpt.objective_examples.experiments2d.sixhumpcamel()
f_sim = GPyOpt.objective_examples.experiments2d.sixhumpcamel(sd = 0.1)
bounds =[{'name': 'var_1', 'type': 'continuous', 'domain': f_true.bounds[0]},
         {'name': 'var_2', 'type': 'continuous', 'domain': f_true.bounds[1]}]
f_true.plot()

# Creates three identical objects that we will later use to compare the optimization strategies 
myBopt2D = GPyOpt.methods.BayesianOptimization(f_sim.f,
                                              domain=bounds,
                                              model_type = 'GP',
                                              acquisition_type='LCB',  
                                              normalize_Y = True,
                                              acquisition_weight = 2)    

# runs the optimization for the three methods
max_iter = 40  # maximum time 40 iterations
max_time = 60  # maximum time 60 seconds

myBopt2D.run_optimization(max_iter,max_time,verbosity=False)            

myBopt2D.plot_acquisition() 

myBopt2D.plot_convergence()

