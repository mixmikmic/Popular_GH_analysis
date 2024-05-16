import os,sys
import numpy
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
sys.path.insert(0,'../')
from utils.mkdesign import create_design_singlecondition
from nipy.modalities.fmri.hemodynamic_models import spm_hrf,compute_regressor
from statsmodels.tsa.arima_process import arma_generate_sample
import scipy.stats
import pymc3

tslength=300
d,design=create_design_singlecondition(blockiness=1.0,deslength=tslength,
                                       blocklength=20,offset=20)
regressor,_=compute_regressor(design,'spm',numpy.arange(0,tslength))


ar1_noise=arma_generate_sample([1,0.3],[1,0.],len(regressor))

X=numpy.hstack((regressor,numpy.ones((len(regressor),1))))
beta=numpy.array([4,100])
noise_sd=10
data = X.dot(beta) + ar1_noise*noise_sd

beta_hat=numpy.linalg.inv(X.T.dot(X)).dot(X.T).dot(data)
resid=data - X.dot(beta_hat)
df=(X.shape[0] - X.shape[1])
mse=resid.dot(resid)
sigma2hat=(mse)/float(df)

xvar=X[:,0].dot(X[:,0])
c=numpy.array([1,0])  # contrast for PPI
t=c.dot(beta_hat)/numpy.sqrt(c.dot(numpy.linalg.inv(X.T.dot(X)).dot(c))*sigma2hat)
print ('betas [slope,intercept]:',beta_hat)
print ('t [for slope vs. zero]=',t, 'p=',1.0 - scipy.stats.t.cdf(t,X.shape[0] - X.shape[1]))


confs = [[beta_hat[0] - scipy.stats.t.ppf(0.975,df) * numpy.sqrt(sigma2hat/xvar), 
         beta_hat[0] + scipy.stats.t.ppf(0.975,df) * numpy.sqrt(sigma2hat/xvar)],
         [beta_hat[1] - scipy.stats.t.ppf(0.975,df) * numpy.sqrt(sigma2hat/X.shape[0]), 
         beta_hat[1] + scipy.stats.t.ppf(0.975,df) * numpy.sqrt(sigma2hat/X.shape[0])]]

print ('slope:',confs[0])
print ('intercept:',confs[1])

prior_sd=10
v=numpy.identity(2)*(prior_sd**2)

beta_hat_bayes=numpy.linalg.inv(X.T.dot(X) + (sigma2hat/(prior_sd**2))*numpy.identity(2)).dot(X.T.dot(data))
print ('betas [slope,intercept]:',beta_hat_bayes)

with pymc3.Model() as model: # model specifications in PyMC3 are wrapped in a with-statement
    # Define priors
    sigma = pymc3.HalfCauchy('sigma', beta=10, testval=1.)
    intercept = pymc3.Normal('Intercept', 0, sd=prior_sd)
    x_coeff = pymc3.Normal('x', 0, sd=prior_sd)
    
    # Define likelihood
    likelihood = pymc3.Normal('y', mu=intercept + x_coeff * X[:,0], 
                        sd=sigma, observed=data)
    
    # Inference!
    start = pymc3.find_MAP() # Find starting value by optimization
    step = pymc3.NUTS(scaling=start) # Instantiate MCMC sampling algorithm
    trace = pymc3.sample(4000, step, start=start, progressbar=False) # draw 2000 posterior samples using NUTS sampling

print(start)

plt.figure(figsize=(7, 7))
pymc3.traceplot(trace[100::5],'x')
plt.tight_layout();
pymc3.autocorrplot(trace[100::5])

pymc3.summary(trace[100:])



