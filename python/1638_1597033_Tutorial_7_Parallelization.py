get_ipython().run_line_magic('pylab', 'inline')
from sklearn.mixture import GaussianMixture
from pomegranate import *
import seaborn, time
seaborn.set_style('whitegrid')

def create_dataset(n_samples, n_dim, n_classes, alpha=1):
    """Create a random dataset with n_samples in each class."""
    
    X = numpy.concatenate([numpy.random.normal(i*alpha, 1, size=(n_samples, n_dim)) for i in range(n_classes)])
    y = numpy.concatenate([numpy.zeros(n_samples) + i for i in range(n_classes)])
    idx = numpy.arange(X.shape[0])
    numpy.random.shuffle(idx)
    return X[idx], y[idx]

n, d, k = 1000000, 5, 3
X, y = create_dataset(n, d, k)

print "sklearn GMM"
get_ipython().run_line_magic('timeit', "GaussianMixture(n_components=k, covariance_type='full', max_iter=15, tol=1e-10).fit(X)")
print 
print "pomegranate GMM"
get_ipython().run_line_magic('timeit', 'GeneralMixtureModel.from_samples(MultivariateGaussianDistribution, k, X, max_iterations=15, stop_threshold=1e-10)')
print
print "pomegranate GMM (4 jobs)"
get_ipython().run_line_magic('timeit', 'GeneralMixtureModel.from_samples(MultivariateGaussianDistribution, k, X, n_jobs=4, max_iterations=15, stop_threshold=1e-10)')

d, k = 25, 2
X, y = create_dataset(1000, d, k)
a = GaussianMixture(k, n_init=1, max_iter=25).fit(X)
b = GeneralMixtureModel.from_samples(MultivariateGaussianDistribution, k, X, max_iterations=25)

del X, y
n = 1000000
X, y = create_dataset(n, d, k)

print "sklearn GMM"
get_ipython().run_line_magic('timeit', '-n 1 a.predict_proba(X)')
print
print "pomegranate GMM"
get_ipython().run_line_magic('timeit', '-n 1 b.predict_proba(X)')
print
print "pomegranate GMM (4 jobs)"
get_ipython().run_line_magic('timeit', '-n 1 b.predict_proba(X, n_jobs=4)')

print (b.predict_proba(X) - b.predict_proba(X, n_jobs=4)).sum()

d, k = 2, 2
X, y = create_dataset(1000, d, k, alpha=2)
a = GaussianMixture(k, n_init=1, max_iter=25).fit(X)
b = GeneralMixtureModel.from_samples(MultivariateGaussianDistribution, k, X, max_iterations=25)

y1, y2 = a.predict(X), b.predict(X)

plt.figure(figsize=(16,6))
plt.subplot(121)
plt.title("sklearn clusters", fontsize=14)
plt.scatter(X[y1==0, 0], X[y1==0, 1], color='m', edgecolor='m')
plt.scatter(X[y1==1, 0], X[y1==1, 1], color='c', edgecolor='c')

plt.subplot(122)
plt.title("pomegranate clusters", fontsize=14)
plt.scatter(X[y2==0, 0], X[y2==0, 1], color='m', edgecolor='m')
plt.scatter(X[y2==1, 0], X[y2==1, 1], color='c', edgecolor='c')

X = numpy.random.randn(1000, 500, 50)

print "pomegranate Gaussian HMM (1 job)"
get_ipython().run_line_magic('timeit', '-n 1 -r 1 HiddenMarkovModel.from_samples(NormalDistribution, 5, X, max_iterations=5)')
print
print "pomegranate Gaussian HMM (2 jobs)"
get_ipython().run_line_magic('timeit', '-n 1 -r 1 HiddenMarkovModel.from_samples(NormalDistribution, 5, X, max_iterations=5, n_jobs=2)')
print
print "pomegranate Gaussian HMM (2 jobs)"
get_ipython().run_line_magic('timeit', '-n 1 -r 1 HiddenMarkovModel.from_samples(NormalDistribution, 5, X, max_iterations=5, n_jobs=4)')

model = HiddenMarkovModel.from_samples(NormalDistribution, 5, X, max_iterations=2, verbose=False)

print "pomegranate Gaussian HMM (1 job)"
get_ipython().run_line_magic('timeit', 'predict_proba(model, X)')
print
print "pomegranate Gaussian HMM (2 jobs)"
get_ipython().run_line_magic('timeit', 'predict_proba(model, X, n_jobs=2)')

def create_model(mus):
    n = mus.shape[0]
    
    starts = numpy.zeros(n)
    starts[0] = 1.
    
    ends = numpy.zeros(n)
    ends[-1] = 0.5
    
    transition_matrix = numpy.zeros((n, n))
    distributions = []
    
    for i in range(n):
        transition_matrix[i, i] = 0.5
        
        if i < n - 1:
            transition_matrix[i, i+1] = 0.5
    
        distribution = IndependentComponentsDistribution([NormalDistribution(mu, 1) for mu in mus[i]])
        distributions.append(distribution)
    
    model = HiddenMarkovModel.from_matrix(transition_matrix, distributions, starts, ends)
    return model
    

def create_mixture(mus):
    hmms = [create_model(mu) for mu in mus]
    return GeneralMixtureModel(hmms)

n, d = 50, 10
mus = [(numpy.random.randn(d, n)*0.2 + numpy.random.randn(n)*2).T for i in range(2)]

model = create_mixture(mus)
X = numpy.random.randn(400, 150, d)

print "pomegranate Mixture of Gaussian HMMs (1 job)"
get_ipython().run_line_magic('timeit', 'model.fit(X, max_iterations=5)')
print

model = create_mixture(mus)
print "pomegranate Mixture of Gaussian HMMs (2 jobs)"
get_ipython().run_line_magic('timeit', 'model.fit(X, max_iterations=5, n_jobs=2)')

model = create_mixture(mus)

print "pomegranate Mixture of Gaussian HMMs (1 job)"
get_ipython().run_line_magic('timeit', 'model.predict_proba(X)')
print

model = create_mixture(mus)
print "pomegranate Mixture of Gaussian HMMs (2 jobs)"
get_ipython().run_line_magic('timeit', 'model.predict_proba(X, n_jobs=2)')

