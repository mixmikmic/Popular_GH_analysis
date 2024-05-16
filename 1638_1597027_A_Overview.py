get_ipython().run_line_magic('matplotlib', 'inline')
import time
import pandas
import random
import numpy
import matplotlib.pyplot as plt
import seaborn; seaborn.set_style('whitegrid')
import itertools

from pomegranate import *

random.seed(0)
numpy.random.seed(0)
numpy.set_printoptions(suppress=True)

get_ipython().run_line_magic('load_ext', 'watermark')
get_ipython().run_line_magic('watermark', '-m -n -p numpy,scipy,pomegranate')

model = NormalDistribution(5, 1)

print(model.probability([4., 6., 7.]))
print(model.log_probability([4., 6., 7.]))

model.sample(n=10)

X = numpy.random.normal(7, 2, size=(100,))

model.fit(X)
model

X = numpy.random.normal(8, 1.5, size=(100,))

model.summarize(X)
model

model.from_summaries()
model

X = numpy.random.normal(3, 0.2, size=(100,))

model.summarize(X)
model.clear_summaries()
model.from_summaries()
model

X = numpy.random.normal(6, 1, size=(250, 1))

model = NormalDistribution.from_samples(X)
model

model = GeneralMixtureModel.from_samples(NormalDistribution, 3, X)
model

print(model.to_json())

model = NormalDistribution(5, 2)

model2 = Distribution.from_json(model.to_json())
model2

d1 = ExponentialDistribution(5.0)
d2 = ExponentialDistribution(0.3)

model = GeneralMixtureModel([d1, d2])
model

X = numpy.random.exponential(3, size=(10,1))

model.predict(X)

model.predict_proba(X)

model.predict_log_proba(X)

X = numpy.random.normal(5, 1, size=(100, 2))
X[50:] += 1

y = numpy.zeros(100)
y[50:] = 1

model1 = NaiveBayes.from_samples(NormalDistribution, X, y)
model2 = NaiveBayes.from_samples(LogNormalDistribution, X, y)

mu = numpy.random.normal(7, 2, size=1000)
std = numpy.random.lognormal(-0.8, 0.8, size=1000)
dur = numpy.random.exponential(50, size=1000)

data = numpy.concatenate([numpy.random.normal(mu_, std_, int(t)) for mu_, std_, t in zip(mu, std, dur)])

plt.figure(figsize=(14, 4))
plt.title("Randomly Generated Signal", fontsize=16)
plt.plot(data)
plt.xlabel("Time", fontsize=14)
plt.ylabel("Signal", fontsize=14)
plt.xlim(0, 3000)
plt.show()

plt.figure(figsize=(12, 3))
plt.subplot(131)
plt.title("mu", fontsize=14)
plt.hist(mu, bins=numpy.arange(0, 15))

plt.subplot(132)
plt.title("sigma", fontsize=14)
plt.hist(std, bins=numpy.arange(0.00, 1.75, 0.05))

plt.subplot(133)
plt.title("Duration", fontsize=14)
plt.hist(dur, bins=numpy.arange(0, 150, 10))
plt.show()

X1 = numpy.array([numpy.random.normal(7, 2, size=400),
                  numpy.random.lognormal(-0.8, 0.8, size=400),
                  numpy.random.exponential(50, size=400)]).T

X2 = numpy.array([numpy.random.normal(8, 2, size=600),
                  numpy.random.lognormal(-1.2, 0.6, size=600),
                  numpy.random.exponential(100, size=600)]).T

X = numpy.concatenate([X1, X2])
y = numpy.zeros(1000)
y[400:] = 1

NaiveBayes.from_samples([NormalDistribution, LogNormalDistribution, ExponentialDistribution], X, y)

X = numpy.concatenate([numpy.random.normal((5, 1), 1, size=(200, 2)),
                       numpy.random.normal((6, 4), 1, size=(200, 2)),
                       numpy.random.normal((3, 5), 1, size=(350, 2)),
                       numpy.random.normal((7, 6), 1, size=(250, 2))])

y = numpy.zeros(1000)
y[400:] = 1

model = BayesClassifier.from_samples(MultivariateGaussianDistribution, X, y)
print model.log_probability(X).sum()


d1 = GeneralMixtureModel.from_samples(MultivariateGaussianDistribution, 2, X[y == 0])
d2 = GeneralMixtureModel.from_samples(MultivariateGaussianDistribution, 2, X[y == 1])
model2 = BayesClassifier([d1, d2], [0.4, 0.6])
print model2.log_probability(X).sum()

X = numpy.random.normal(4, 1, size=1000)

get_ipython().run_line_magic('timeit', '-n 1 -r 1 numpy.mean(X), numpy.std(X)')
get_ipython().run_line_magic('timeit', '-n 1 -r 1 NormalDistribution.from_samples(X)')

X = numpy.random.normal(4, 1, size=10000000)

get_ipython().run_line_magic('timeit', 'numpy.mean(X), numpy.std(X)')
get_ipython().run_line_magic('timeit', 'NormalDistribution.from_samples(X)')

X = numpy.random.normal(4, 1, size=(1000000, 3))

get_ipython().run_line_magic('timeit', 'numpy.mean(X, axis=0), numpy.cov(X, rowvar=False, ddof=0)')
get_ipython().run_line_magic('timeit', 'MultivariateGaussianDistribution.from_samples(X)')

X = numpy.random.normal(4, 1, size=(100000, 1000))

get_ipython().run_line_magic('timeit', 'numpy.mean(X, axis=0), numpy.cov(X, rowvar=False, ddof=0)')
get_ipython().run_line_magic('timeit', 'MultivariateGaussianDistribution.from_samples(X)')

from scipy.stats import norm

d = NormalDistribution(0, 1)
x = numpy.random.normal(0, 1, size=(10000000,))

get_ipython().run_line_magic('timeit', 'norm.logpdf(x, 0, 1)')
get_ipython().run_line_magic('timeit', 'NormalDistribution(0, 1).log_probability(x)')

print "\nlogp difference: {}".format((norm.logpdf(x, 0, 1) - NormalDistribution(0, 1).log_probability(x)).sum())

from scipy.stats import multivariate_normal

dim = 2500
n = 1000

mu = numpy.random.normal(6, 1, size=dim)
cov = numpy.eye(dim)

X = numpy.random.normal(8, 1, size=(n, dim))

d = MultivariateGaussianDistribution(mu, cov)

get_ipython().run_line_magic('timeit', 'multivariate_normal.logpdf(X, mu, cov)')
get_ipython().run_line_magic('timeit', 'MultivariateGaussianDistribution(mu, cov).log_probability(X)')
get_ipython().run_line_magic('timeit', 'd.log_probability(X)')

print "\nlogp difference: {}".format((multivariate_normal.logpdf(X, mu, cov) - d.log_probability(X)).sum())

from sklearn.mixture import GaussianMixture

X = numpy.random.normal(8, 1, size=(10000, 100))

get_ipython().run_line_magic('timeit', 'model1 = GaussianMixture(5, max_iter=10).fit(X)')
get_ipython().run_line_magic('timeit', 'model2 = GeneralMixtureModel.from_samples(MultivariateGaussianDistribution, 5, X, max_iterations=10)')

from sklearn.naive_bayes import GaussianNB

X = numpy.random.normal(8, 1, size=(100000, 500))
X[:50000] += 1

y = numpy.zeros(100000)
y[:50000] = 1

get_ipython().run_line_magic('timeit', 'GaussianNB().fit(X, y)')
get_ipython().run_line_magic('timeit', 'NaiveBayes.from_samples(NormalDistribution, X, y)')

model = NaiveBayes.from_samples(LogNormalDistribution, X, y)

