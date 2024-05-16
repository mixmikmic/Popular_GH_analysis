get_ipython().run_line_magic('pylab', 'inline')
from pomegranate import *
import seaborn
seaborn.set_style('whitegrid')
numpy.random.seed(0)

X = numpy.concatenate([numpy.random.normal(0, 1, size=(1000)), numpy.random.normal(6, 1, size=(1250))])

plt.title("Bimodal Distribution", fontsize=14)
plt.hist(X, bins=numpy.arange(-3, 9, 0.1), alpha=0.6)
plt.ylabel("Count", fontsize=14)
plt.yticks(fontsize=12)
plt.xlabel("Value", fontsize=14)
plt.yticks(fontsize=12)
plt.vlines(numpy.mean(X), 0, 80, color='r', label="Mean")
plt.vlines(numpy.median(X), 0, 80, color='b', label="Median")
plt.legend(fontsize=14)
plt.show()

X = numpy.concatenate([X, [numpy.nan]*500])
X_imp = X.copy()
X_imp[numpy.isnan(X_imp)] = numpy.mean(X_imp[~numpy.isnan(X_imp)])

plt.title("Bimodal Distribution", fontsize=14)
plt.hist(X_imp, bins=numpy.arange(-3, 9, 0.1), alpha=0.6)
plt.ylabel("Count", fontsize=14)
plt.yticks(fontsize=12)
plt.xlabel("Value", fontsize=14)
plt.yticks(fontsize=12)
plt.vlines(numpy.mean(X), 0, 80, color='r', label="Mean")
plt.vlines(numpy.median(X), 0, 80, color='b', label="Median")
plt.legend(fontsize=14)
plt.show()

x = numpy.arange(-3, 9, 0.1)
model1 = GeneralMixtureModel.from_samples(NormalDistribution, 2, X_imp.reshape(X_imp.shape[0], 1))
model2 = GeneralMixtureModel.from_samples(NormalDistribution, 2, X.reshape(X.shape[0], 1))
p1 = model1.probability(x.reshape(x.shape[0], 1))
p2 = model2.probability(x.reshape(x.shape[0], 1))

plt.figure(figsize=(12, 3))
plt.subplot(121)
plt.title("Mean Impute Missing Values", fontsize=14)
plt.hist(X_imp, bins=x, alpha=0.6, density=True)
plt.plot(x, p1, color='b')
plt.ylabel("Count", fontsize=14); plt.yticks(fontsize=12)
plt.xlabel("Value", fontsize=14); plt.yticks(fontsize=12)

plt.subplot(122)
plt.title("Ignore Missing Values", fontsize=14)
plt.hist(X[~numpy.isnan(X)], bins=x, alpha=0.6, density=True)
plt.plot(x, p2, color='b')
plt.ylabel("Count", fontsize=14); plt.yticks(fontsize=12)
plt.xlabel("Value", fontsize=14); plt.yticks(fontsize=12)
plt.show()

X = numpy.concatenate([numpy.random.normal(0, 1, size=(750)), [numpy.nan]*250])
X_imp = X.copy()
X_imp[numpy.isnan(X_imp)] = numpy.mean(X_imp[~numpy.isnan(X_imp)])

x = numpy.arange(-3, 3, 0.1)
d1 = NormalDistribution.from_samples(X_imp)
d2 = NormalDistribution.from_samples(X)
p1 = d1.probability(x.reshape(x.shape[0], 1))
p2 = d2.probability(x.reshape(x.shape[0], 1))

plt.figure(figsize=(12, 3))
plt.subplot(121)
plt.title("Mean Impute Missing Values", fontsize=14)
plt.hist(X_imp, bins=x, alpha=0.6, density=True, label="$\sigma$ = {:4.4}".format(d1.parameters[1]))
plt.plot(x, p1, color='b')
plt.ylabel("Count", fontsize=14); plt.yticks(fontsize=12)
plt.xlabel("Value", fontsize=14); plt.yticks(fontsize=12)
plt.legend(fontsize=14)

plt.subplot(122)
plt.title("Ignore Missing Values", fontsize=14)
plt.hist(X[~numpy.isnan(X)], bins=x, alpha=0.6, density=True, label="$\sigma$ = {:4.4}".format(d2.parameters[1]))
plt.plot(x, p2, color='b')
plt.ylabel("Count", fontsize=14); plt.yticks(fontsize=12)
plt.xlabel("Value", fontsize=14); plt.yticks(fontsize=12)
plt.legend(fontsize=14)
plt.show()

n, d, steps = 1000, 10, 50
diffs1 = numpy.zeros(int(steps*0.86))
diffs2 = numpy.zeros(int(steps*0.86))

X = numpy.random.normal(6, 3, size=(n, d))

for k, size in enumerate(range(0, int(n*d*0.86), n*d / steps)):
    idxs = numpy.random.choice(numpy.arange(n*d), replace=False, size=size)
    i, j = idxs / d, idxs % d

    cov_true = numpy.cov(X, rowvar=False, bias=True)
    X_nan = X.copy()
    X_nan[i, j] = numpy.nan

    X_mean = X_nan.copy()
    for col in range(d):
        mask = numpy.isnan(X_mean[:,col])
        X_mean[mask, col] = X_mean[~mask, col].mean()

    diff = numpy.abs(numpy.cov(X_mean, rowvar=False, bias=True) - cov_true).sum()
    diffs1[k] = diff

    dist = MultivariateGaussianDistribution.from_samples(X_nan)
    diff = numpy.abs(numpy.array(dist.parameters[1]) - cov_true).sum()
    diffs2[k] = diff

plt.title("Error in Multivariate Gaussian Covariance Matrix", fontsize=16)
plt.plot(diffs1, label="Mean")
plt.plot(diffs2, label="Ignore")

plt.xlabel("Percentage Missing", fontsize=14)
plt.ylabel("L1 Errors", fontsize=14)
plt.xticks(range(0, 51, 10), numpy.arange(0, 5001, 1000) / 5000.)
plt.xlim(0, 50)
plt.legend(fontsize=14)
plt.show()

X = numpy.random.randn(100)
X_nan = numpy.concatenate([X, [numpy.nan]*100])

print "Fitting only to observed values:"
print NormalDistribution.from_samples(X)
print 
print "Fitting to observed and missing values:"
print NormalDistribution.from_samples(X_nan)

X = numpy.random.normal(0, 1, size=(500, 3))
idxs = numpy.random.choice(1500, replace=False, size=500)
i, j = idxs // 3, idxs % 3
X[i, j] = numpy.nan

d = IndependentComponentsDistribution.from_samples(X, distributions=[NormalDistribution]*3)
d

NormalDistribution(1, 2).probability(numpy.nan)

d.probability((numpy.nan, 2, 3))

d.distributions[1].probability(2) * d.distributions[2].probability(3)

X = numpy.concatenate([numpy.random.normal(0, 1, size=(50, 2)), numpy.random.normal(3, 1, size=(75, 2))])
X_nan = X.copy()

idxs = numpy.random.choice(250, replace=False, size=50)
i, j = idxs // 2, idxs % 2
X_nan[i, j] = numpy.nan

model1 = Kmeans.from_samples(2, X)
model2 = Kmeans.from_samples(2, X_nan)

y1 = model1.predict(X)
y2 = model2.predict(X_nan)

plt.figure(figsize=(14, 6))
plt.subplot(121)
plt.title("Fit w/o Missing Values", fontsize=16)
plt.scatter(X[y1 == 0,0], X[y1 == 0,1], color='b')
plt.scatter(X[y1 == 1,0], X[y1 == 1,1], color='r')

plt.subplot(122)
plt.title("Fit w/ Missing Values", fontsize=16)
plt.scatter(X[y2 == 0,0], X[y2 == 0,1], color='b')
plt.scatter(X[y2 == 1,0], X[y2 == 1,1], color='r')
plt.show()

X = numpy.concatenate([numpy.random.normal(0, 1, size=(1000, 10)), numpy.random.normal(2, 1, size=(1250, 10))])

idxs = numpy.random.choice(22500, replace=False, size=5000)
i, j = idxs // 10, idxs % 10

X_nan = X.copy()
X_nan[i, j] = numpy.nan

get_ipython().run_line_magic('timeit', 'GeneralMixtureModel.from_samples(MultivariateGaussianDistribution, 2, X, max_iterations=10)')
get_ipython().run_line_magic('timeit', 'GeneralMixtureModel.from_samples(MultivariateGaussianDistribution, 2, X_nan, max_iterations=10)')

get_ipython().run_line_magic('timeit', '-n 100 GeneralMixtureModel.from_samples([NormalDistribution]*2, 2, X, max_iterations=10)')
get_ipython().run_line_magic('timeit', '-n 100 GeneralMixtureModel.from_samples([NormalDistribution]*2, 2, X_nan, max_iterations=10)')

y = numpy.concatenate([numpy.zeros(1000), numpy.ones(1250)])

get_ipython().run_line_magic('timeit', '-n 100 BayesClassifier.from_samples(MultivariateGaussianDistribution, X, y)')
get_ipython().run_line_magic('timeit', '-n 100 BayesClassifier.from_samples(MultivariateGaussianDistribution, X_nan, y)')

idx = numpy.random.choice(2250, replace=False, size=750)
y_nan = y.copy()
y_nan[idx] = -1

model = BayesClassifier.from_samples(MultivariateGaussianDistribution, X_nan, y_nan, verbose=True)

get_ipython().run_line_magic('timeit', 'BayesClassifier.from_samples(MultivariateGaussianDistribution, X, y_nan)')
get_ipython().run_line_magic('timeit', 'BayesClassifier.from_samples(MultivariateGaussianDistribution, X_nan, y_nan)')

d1 = DiscreteDistribution({'A': 0.25, 'B': 0.75})
d2 = DiscreteDistribution({'A': 0.67, 'B': 0.33})

s1 = State(d1, name="s1")
s2 = State(d2, name="s2")

model = HiddenMarkovModel()
model.add_states(s1, s2)
model.add_transition(model.start, s1, 1.0)
model.add_transition(s1, s1, 0.5)
model.add_transition(s1, s2, 0.5)
model.add_transition(s2, s2, 0.5)
model.add_transition(s2, s1, 0.5)
model.bake()

numpy.exp(model.forward(['A', 'B', 'A', 'A']))

numpy.exp(model.forward(['A', 'nan', 'A', 'A']))

model.predict(['A', 'A', 'B', 'B', 'A', 'A'])

model.predict(['A', 'nan', 'B', 'B', 'nan', 'A'])

X = numpy.random.randint(3, size=(500, 10)).astype('float64')

idxs = numpy.random.choice(5000, replace=False, size=2000)
i, j = idxs // 10, idxs % 10
X_nan = X.copy()
X_nan[i, j] = numpy.nan

get_ipython().run_line_magic('timeit', "-n 100 BayesianNetwork.from_samples(X, algorithm='exact')")
get_ipython().run_line_magic('timeit', "-n 100 BayesianNetwork.from_samples(X_nan, algorithm='exact')")

