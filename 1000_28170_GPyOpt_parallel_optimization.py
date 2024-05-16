from IPython.display import HTML 
HTML('<iframe src=http://arxiv.org/pdf/1505.08052v4.pdf width=700 height=550></iframe>')

get_ipython().magic('pylab inline')
import GPyOpt

# --- Objective function
objective_true  = GPyOpt.objective_examples.experiments2d.branin()                 # true function
objective_noisy = GPyOpt.objective_examples.experiments2d.branin(sd = 0.1)         # noisy version
bounds = objective_noisy.bounds        

domain = [{'name': 'var_1', 'type': 'continuous', 'domain': bounds[0]}, ## use default bounds
          {'name': 'var_2', 'type': 'continuous', 'domain': bounds[1]}]

objective_true.plot()

batch_size = 4
num_cores = 4

from numpy.random import seed
seed(123)
BO_demo_parallel = GPyOpt.methods.BayesianOptimization(f=objective_noisy.f,  
                                            domain = domain,                  
                                            acquisition_type = 'EI',              
                                            normalize_Y = True,
                                            initial_design_numdata = 10,
                                            evaluator_type = 'local_penalization',
                                            batch_size = batch_size,
                                            num_cores = num_cores,
                                            acquisition_jitter = 0)    

# --- Run the optimization for 10 iterations
max_iter = 10                                                     
BO_demo_parallel.run_optimization(max_iter)

BO_demo_parallel.plot_acquisition()

BO_demo_parallel.suggested_sample

objective_noisy.min

