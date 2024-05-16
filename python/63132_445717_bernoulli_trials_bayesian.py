get_ipython().magic('matplotlib inline')
import numpy as np
import pandas as pd
import statsmodels.api as sm
import sympy as sp
import pymc
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats
from scipy.special import gamma

from sympy.interactive import printing
printing.init_printing()

# Simulate data
np.random.seed(123)

nobs = 100
theta = 0.3
Y = np.random.binomial(1, theta, nobs)

# Plot the data
fig = plt.figure(figsize=(7,3))
gs = gridspec.GridSpec(1, 2, width_ratios=[5, 1]) 
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])

ax1.plot(range(nobs), Y, 'x')
ax2.hist(-Y, bins=2)

ax1.yaxis.set(ticks=(0,1), ticklabels=('Failure', 'Success'))
ax2.xaxis.set(ticks=(-1,0), ticklabels=('Success', 'Failure'));

ax1.set(title=r'Bernoulli Trial Outcomes $(\theta=0.3)$', xlabel='Trial', ylim=(-0.2, 1.2))
ax2.set(ylabel='Frequency')

fig.tight_layout()

t, T, s = sp.symbols('theta, T, s')

# Create the function symbolically
likelihood = (t**s)*(1-t)**(T-s)

# Convert it to a Numpy-callable function
_likelihood = sp.lambdify((t,T,s), likelihood, modules='numpy')

# For alpha_1 = alpha_2 = 1, the Beta distribution
# degenerates to a uniform distribution
a1 = 1
a2 = 1

# Prior Mean
prior_mean = a1 / (a1 + a2)
print 'Prior mean:', prior_mean

# Plot the prior
fig = plt.figure(figsize=(10,4))
ax = fig.add_subplot(111)
X = np.linspace(0,1, 1000)
ax.plot(X, stats.beta(a1, a2).pdf(X), 'g');

# Cleanup
ax.set(title='Prior Distribution', ylim=(0,12))
ax.legend(['Prior']);

# Find the hyperparameters of the posterior
a1_hat = a1 + Y.sum()
a2_hat = a2 + nobs - Y.sum()

# Posterior Mean
post_mean = a1_hat / (a1_hat + a2_hat)
print 'Posterior Mean (Analytic):', post_mean

# Plot the analytic posterior
fig = plt.figure(figsize=(10,4))
ax = fig.add_subplot(111)
X = np.linspace(0,1, 1000)
ax.plot(X, stats.beta(a1_hat, a2_hat).pdf(X), 'r');

# Plot the prior
ax.plot(X, stats.beta(a1, a2).pdf(X), 'g');

# Cleanup
ax.set(title='Posterior Distribution (Analytic)', ylim=(0,12))
ax.legend(['Posterior (Analytic)', 'Prior']);

#%%timeit
print 'Timing: 1 loops, best of 3: 356 ms per loop'

# Metropolis-Hastings parameters
G1 = 1000 # burn-in period
G = 10000 # draws from the (converged) posterior

# Model parameters
sigma = 0.1
thetas = [0.5]             # initial value for theta
etas = np.random.normal(0, sigma, G1+G) # random walk errors
unif = np.random.uniform(size=G1+G)     # comparators for accept_probs

# Callable functions for likelihood and prior
prior_const = gamma(a1) * gamma(a2) / gamma(a1 + a2)
mh_ll = lambda theta: _likelihood(theta, nobs, Y.sum())
def mh_prior(theta):
    prior = 0
    if theta >= 0 and theta <= 1:
        prior = prior_const*(theta**(a1-1))*((1-theta)**(a2-1))
    return prior
mh_accept = lambda theta: mh_ll(theta) * mh_prior(theta)

theta_prob = mh_accept(thetas[-1])

# Metropolis-Hastings iterations
for i in range(G1+G):
    # Draw theta
    
    # Generate the proposal
    theta = thetas[-1]
    theta_star = theta + etas[i]
    theta_star_prob = mh_accept(theta_star)
    # Calculate the acceptance probability
    accept_prob = theta_star_prob / theta_prob
    
    # Append the new draw
    if accept_prob > unif[i]:
        theta = theta_star
        theta_prob = theta_star_prob
    thetas.append(theta)

# Posterior Mean
print 'Posterior Mean (MH):', np.mean(thetas[G1:])

# Plot the posterior
fig = plt.figure(figsize=(10,4))
ax = fig.add_subplot(111)
# Plot MH draws
ax.hist(thetas[G1:], bins=50, normed=True);
# Plot analytic posterior
X = np.linspace(0,1, 1000)
ax.plot(X, stats.beta(a1_hat, a2_hat).pdf(X), 'r');
# Plot prior
ax.plot(X, stats.beta(a1, a2).pdf(X), 'g')

# Cleanup
ax.set(title='Metropolis-Hastings via pure Python (10,000 Draws; 1,000 Burned)', ylim=(0,12))
ax.legend(['Posterior (Analytic)', 'Prior', 'Posterior Draws (MH)']);

get_ipython().magic('load_ext cythonmagic')

get_ipython().run_cell_magic('cython', '', '\nimport numpy as np\nfrom scipy.special import gamma\ncimport numpy as np\ncimport cython\n\nfrom libc.math cimport pow\n\ncdef double likelihood(double theta, int T, int s):\n    return pow(theta, s)*pow(1-theta, T-s)\n\ncdef double prior(double theta, double a1, double a2, double prior_const):\n    if theta < 0 or theta > 1:\n        return 0\n    return prior_const*pow(theta, a1-1)*pow(1-theta, a2-1)\n\ncdef np.ndarray[np.float64_t, ndim=1] draw_posterior(np.ndarray[np.float64_t, ndim=1] theta, double eta, double unif, int T, int s, double a1, double a2, double prior_const):\n    cdef double theta_star, theta_star_prob, accept_prob\n    \n    theta_star = theta[0] + eta\n    theta_star_prob = likelihood(theta_star, T, s) * prior(theta_star, a1, a2, prior_const)\n    \n    accept_prob = theta_star_prob / theta[1]\n    \n    if accept_prob > unif:\n        theta[0] = theta_star\n        theta[1] = theta_star_prob\n        \n    return theta\n\ndef mh(double theta_init, int T, int s, double sigma, double a1, double a2, int G1, int G):\n    \n    cdef np.ndarray[np.float64_t, ndim = 1] theta, thetas, etas, unif\n    cdef double prior_const, theta_prob\n    cdef int t\n    \n    prior_const = gamma(a1) * gamma(a2) / gamma(a1 + a2)\n    theta_prob = likelihood(theta_init, T, s) * prior(theta_init, a1, a2, prior_const)\n    \n    theta = np.array([theta_init, theta_prob])\n    \n    thetas = np.zeros((G1+G,))\n    etas = np.random.normal(0, sigma, G1+G)\n    unif = np.random.uniform(size=G1+G)\n    \n    for t in range(G1+G):\n        theta = draw_posterior(theta, etas[t], unif[t], T, s, a1, a2, prior_const)\n        thetas[t] = theta[0]\n        \n    return thetas')

#%%timeit
print 'Timing: 10 loops, best of 3: 20.7 ms per loop'
thetas = mh(0.5, nobs, Y.sum(), sigma, a1, a2, G1, G)

# Posterior Mean
print 'Posterior Mean (MH):', np.mean(thetas[G1:])

# Plot the posterior
fig = plt.figure(figsize=(10,4))
ax = fig.add_subplot(111)
# Plot MH draws
ax.hist(thetas[G1:], bins=50, normed=True);
# Plot analytic posterior
X = np.linspace(0,1, 1000)
ax.plot(X, stats.beta(a1_hat, a2_hat).pdf(X), 'r');
# Plot prior
ax.plot(X, stats.beta(a1, a2).pdf(X), 'g')

# Cleanup
ax.set(title='Metropolis-Hastings via Cython (10,000 Draws; 1,000 Burned)', ylim=(0,12))
ax.legend(['Posterior (Analytic)', 'Prior', 'Posterior Draws (MH)']);

G1 = 100000
G = 1000000

#%%timeit
print 'Timing: 1 loops, best of 3: 2.09 s per loop'
thetas = mh(0.5, nobs, Y.sum(), sigma, a1, a2, G1, G)

# Posterior Mean
print 'Posterior Mean (MH):', np.mean(thetas[G1:])

# Plot the posterior
fig = plt.figure(figsize=(10,4))
ax = fig.add_subplot(111)
# Plot MH draws
ax.hist(thetas[G1:], bins=50, normed=True);
# Plot analytic posterior
X = np.linspace(0,1, 1000)
ax.plot(X, stats.beta(a1_hat, a2_hat).pdf(X), 'r');
# Plot prior
ax.plot(X, stats.beta(a1, a2).pdf(X), 'g')

# Cleanup
ax.set(title='Metropolis-Hastings via Cython (1,000,000 Draws; 100,000 Burned)', ylim=(0,12))
ax.legend(['Posterior (Analytic)', 'Prior', 'Posterior Draws (MH)']);

G1 = 1000
G = 10000

#%%timeit
print 'Timing: 1 loops, best of 3: 590 ms per loop'

pymc_theta = pymc.Beta('pymc_theta', a1, a2, value=0.5)
pymc_Y = pymc.Bernoulli('pymc_Y', p=pymc_theta, value=Y, observed=True)

model = pymc.MCMC([pymc_theta, pymc_Y])
model.sample(iter=G+G1, burn=G1, progress_bar=False)

model.summary()
thetas = model.trace('pymc_theta')[:]

# Posterior Mean
# (use all of `thetas` b/c PyMC already removed the burn-in runs here)
print 'Posterior Mean (MH):', np.mean(thetas)

# Plot the posterior
fig = plt.figure(figsize=(10,4))
ax = fig.add_subplot(111)
# Plot MH draws
ax.hist(thetas, bins=50, normed=True);
# Plot analytic posterior
X = np.linspace(0,1, 1000)
ax.plot(X, stats.beta(a1_hat, a2_hat).pdf(X), 'r');
# Plot prior
ax.plot(X, stats.beta(a1, a2).pdf(X), 'g')

# Cleanup
ax.set(title='Metropolis-Hastings via PyMC (10,000 Draws; 1,000 Burned)', ylim=(0,12))
ax.legend(['Posterior (Analytic)', 'Prior', 'Posterior Draws (MH)']);

