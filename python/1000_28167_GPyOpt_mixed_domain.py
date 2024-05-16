get_ipython().magic('pylab inline')
import GPyOpt
from numpy.random import seed
seed(123)

func  = GPyOpt.objective_examples.experimentsNd.alpine1(input_dim=5) 

mixed_domain =[{'name': 'var1_2', 'type': 'continuous', 'domain': (-10,10),'dimensionality': 2},
               {'name': 'var3', 'type': 'continuous', 'domain': (-8,3)},
               {'name': 'var4', 'type': 'discrete', 'domain': (-2,0,2)},
               {'name': 'var5', 'type': 'discrete', 'domain': (-1,5)}]

myBopt = GPyOpt.methods.BayesianOptimization(f=func.f,                   # function to optimize       
                                             domain=mixed_domain,        # box-constrains of the problem
                                             initial_design_numdata = 20,# number data initial design
                                             acquisition_type='EI',      # Expected Improvement
                                             exact_feval = True)         # True evaluations

max_iter = 10
max_time = 60

myBopt.run_optimization(max_iter, max_time)

myBopt.X

myBopt.plot_convergence()

myBopt.x_opt

