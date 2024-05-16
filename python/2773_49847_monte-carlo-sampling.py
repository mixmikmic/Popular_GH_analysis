import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import uniform
from scipy.stats import norm
get_ipython().magic('matplotlib inline')

# Gaussian Distribution
x = np.random.normal(5, 1, 10000)
# Law of law numbers -- This mean will be equal to the expected value when the number of samples is large
# Central Limit Theorem -- this mean is Gaussian Distributed.
print('Expected Value of x  = {} with variance = {}'.format(np.mean(x), np.var(x)))
plt.hist(x)
plt.title('Samples drawn from a Gaussian Distribution')
plt.show()

# Uniform Distribution
x = np.random.rand(10000)
# Law of law numbers -- This mean will be equal to the expected value when the number of samples is large
# Central Limit Theorem -- This mean is Gaussian Distributed.
print('Expected Value of x  = {} with variance = {}'.format(np.mean(x), np.var(x)))
plt.hist(x)
plt.title('Samples drawn from a Uniform Distribution')
plt.show()

# SIDE NOTE FOR ME -- `scipy.stats.rv_continuous` is the base class.

# Sample u from a uniform distribution
u = np.random.rand(10000)
# Sample x from the inverse CDF function
# In Scipy it's called ppf
samples = norm.ppf(u, 0, 1)
plt.hist(samples)
plt.title('Samples drawn from a Gaussian Distribution')
plt.show()

# CODE -- TODO

