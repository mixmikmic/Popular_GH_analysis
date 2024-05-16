from pomegranate import *
import seaborn
get_ipython().run_line_magic('pylab', 'inline')
seaborn.set_style('whitegrid')
numpy.set_printoptions(suppress=True)

X = numpy.concatenate((numpy.random.normal(3, 1, 200), numpy.random.normal(10, 2, 1000)))
y = numpy.concatenate((numpy.zeros(200), numpy.ones(1000)))

x1 = X[:200]
x2 = X[200:]

plt.figure(figsize=(16, 5))
plt.hist(x1, bins=25, color='m', edgecolor='m', label="Class A")
plt.hist(x2, bins=25, color='c', edgecolor='c', label="Class B")
plt.xlabel("Value", fontsize=14)
plt.ylabel("Count", fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()

d1 = NormalDistribution.from_samples(x1)
d2 = NormalDistribution.from_samples(x2)
idxs = numpy.arange(0, 15, 0.1)

p1 = map(d1.probability, idxs)
p2 = map(d2.probability, idxs)

plt.figure(figsize=(16, 5))
plt.plot(idxs, p1, color='m'); plt.fill_between(idxs, 0, p1, facecolor='m', alpha=0.2)
plt.plot(idxs, p2, color='c'); plt.fill_between(idxs, 0, p2, facecolor='c', alpha=0.2)
plt.xlabel("Value", fontsize=14)
plt.ylabel("Probability", fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()

magenta_prior = 1. * len(x1) / len(X)
cyan_prior = 1. * len(x2) / len(X)

plt.figure(figsize=(4, 6))
plt.title("Prior Probabilities P(M)", fontsize=14)
plt.bar(0, magenta_prior, facecolor='m', edgecolor='m')
plt.bar(1, cyan_prior, facecolor='c', edgecolor='c')
plt.xticks([0, 1], ['P(Magenta)', 'P(Cyan)'], fontsize=14)
plt.yticks(fontsize=14)
plt.show()

d1 = NormalDistribution.from_samples(x1)
d2 = NormalDistribution.from_samples(x2)
idxs = numpy.arange(0, 15, 0.1)

p_magenta = numpy.array(map(d1.probability, idxs)) * magenta_prior
p_cyan = numpy.array(map(d2.probability, idxs)) * cyan_prior

plt.figure(figsize=(16, 5))
plt.plot(idxs, p_magenta, color='m'); plt.fill_between(idxs, 0, p_magenta, facecolor='m', alpha=0.2)
plt.plot(idxs, p_cyan, color='c'); plt.fill_between(idxs, 0, p_cyan, facecolor='c', alpha=0.2)
plt.xlabel("Value", fontsize=14)
plt.ylabel("P(M)P(D|M)", fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()

magenta_posterior = p_magenta / (p_magenta + p_cyan)
cyan_posterior = p_cyan / (p_magenta + p_cyan)

plt.figure(figsize=(16, 5))
plt.subplot(211)
plt.plot(idxs, p_magenta, color='m'); plt.fill_between(idxs, 0, p_magenta, facecolor='m', alpha=0.2)
plt.plot(idxs, p_cyan, color='c'); plt.fill_between(idxs, 0, p_cyan, facecolor='c', alpha=0.2)
plt.xlabel("Value", fontsize=14)
plt.ylabel("P(M)P(D|M)", fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.subplot(212)
plt.plot(idxs, magenta_posterior, color='m')
plt.plot(idxs, cyan_posterior, color='c')
plt.xlabel("Value", fontsize=14)
plt.ylabel("P(M|D)", fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()

idxs = idxs.reshape(idxs.shape[0], 1)
X = X.reshape(X.shape[0], 1)

model = NaiveBayes.from_samples(NormalDistribution, X, y)
posteriors = model.predict_proba(idxs)

plt.figure(figsize=(14, 4))
plt.plot(idxs, posteriors[:,0], color='m')
plt.plot(idxs, posteriors[:,1], color='c')
plt.xlabel("Value", fontsize=14)
plt.ylabel("P(M|D)", fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()

X = numpy.concatenate([numpy.random.normal(3, 2, size=(150, 2)), numpy.random.normal(7, 1, size=(250, 2))])
y = numpy.concatenate([numpy.zeros(150), numpy.ones(250)])

plt.figure(figsize=(8, 8))
plt.scatter(X[y == 0, 0], X[y == 0, 1], color='c')
plt.scatter(X[y == 1, 0], X[y == 1, 1], color='m')
plt.xlim(-2, 10)
plt.ylim(-4, 12)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()

from sklearn.naive_bayes import GaussianNB

model = NaiveBayes.from_samples(NormalDistribution, X, y)
clf = GaussianNB().fit(X, y)

xx, yy = np.meshgrid(np.arange(-2, 10, 0.02), np.arange(-4, 12, 0.02))
Z1 = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
Z2 = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.figure(figsize=(16, 8))
plt.subplot(121)
plt.title("pomegranate naive Bayes", fontsize=16)
plt.scatter(X[y == 0, 0], X[y == 0, 1], color='c')
plt.scatter(X[y == 1, 0], X[y == 1, 1], color='m')
plt.contour(xx, yy, Z1)
plt.xlim(-2, 10)
plt.ylim(-4, 12)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.subplot(122)
plt.title("sklearn naive Bayes", fontsize=16)
plt.scatter(X[y == 0, 0], X[y == 0, 1], color='c')
plt.scatter(X[y == 1, 0], X[y == 1, 1], color='m')
plt.contour(xx, yy, Z2)
plt.xlim(-2, 10)
plt.ylim(-4, 12)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()

def plot_signal(X, n):
    plt.figure(figsize=(16, 6))
    t_current = 0
    for i in range(n):
        mu, std, t = X[i]
        chunk = numpy.random.normal(mu, std, int(t))
        plt.plot(numpy.arange(t_current, t_current+t), chunk, c='cm'[i % 2])
        t_current += t
    
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel("Time (s)", fontsize=14)
    plt.ylabel("Signal", fontsize=14)
    plt.ylim(20, 40)
    plt.show()

def create_signal(n):
    X, y = [], []
    for i in range(n):
        mu = numpy.random.normal(30.0, 0.4)
        std = numpy.random.lognormal(-0.1, 0.4)
        t = int(numpy.random.exponential(50)) + 1
        X.append([mu, std, int(t)])
        y.append(0)

        mu = numpy.random.normal(30.5, 0.8)
        std = numpy.random.lognormal(-0.3, 0.6)
        t = int(numpy.random.exponential(200)) + 1
        X.append([mu, std, int(t)])
        y.append(1)
    
    return numpy.array(X), numpy.array(y)

X_train, y_train = create_signal(1000)
X_test, y_test = create_signal(250)
plot_signal(X_train, 20)

model = NaiveBayes.from_samples(NormalDistribution, X_train, y_train)
print "Gaussian Naive Bayes: ", (model.predict(X_test) == y_test).mean()

clf = GaussianNB().fit(X_train, y_train)
print "sklearn Gaussian Naive Bayes: ", (clf.predict(X_test) == y_test).mean()

plt.figure(figsize=(14, 4))
plt.subplot(131)
plt.title("Mean")
plt.hist(X_train[y_train == 0, 0], color='c', alpha=0.5, bins=25)
plt.hist(X_train[y_train == 1, 0], color='m', alpha=0.5, bins=25)

plt.subplot(132)
plt.title("Standard Deviation")
plt.hist(X_train[y_train == 0, 1], color='c', alpha=0.5, bins=25)
plt.hist(X_train[y_train == 1, 1], color='m', alpha=0.5, bins=25)

plt.subplot(133)
plt.title("Duration")
plt.hist(X_train[y_train == 0, 2], color='c', alpha=0.5, bins=25)
plt.hist(X_train[y_train == 1, 2], color='m', alpha=0.5, bins=25)
plt.show()

model = NaiveBayes.from_samples(NormalDistribution, X_train, y_train)
print "Gaussian Naive Bayes: ", (model.predict(X_test) == y_test).mean()

clf = GaussianNB().fit(X_train, y_train)
print "sklearn Gaussian Naive Bayes: ", (clf.predict(X_test) == y_test).mean()

model = NaiveBayes.from_samples([NormalDistribution, LogNormalDistribution, ExponentialDistribution], X_train, y_train)
print "Heterogeneous Naive Bayes: ", (model.predict(X_test) == y_test).mean()

get_ipython().run_line_magic('timeit', 'GaussianNB().fit(X_train, y_train)')
get_ipython().run_line_magic('timeit', 'NaiveBayes.from_samples(NormalDistribution, X_train, y_train)')
get_ipython().run_line_magic('timeit', 'NaiveBayes.from_samples([NormalDistribution, LogNormalDistribution, ExponentialDistribution], X_train, y_train)')

pom_time, skl_time = [], []

n1, n2 = 15000, 60000,
for d in range(1, 101, 5): 
    X = numpy.concatenate([numpy.random.normal(3, 2, size=(n1, d)), numpy.random.normal(7, 1, size=(n2, d))])
    y = numpy.concatenate([numpy.zeros(n1), numpy.ones(n2)])

    tic = time.time()
    for i in range(25):
        GaussianNB().fit(X, y)
    skl_time.append((time.time() - tic) / 25)
    
    tic = time.time()
    for i in range(25):
        NaiveBayes.from_samples(NormalDistribution, X, y)
    pom_time.append((time.time() - tic) / 25)

plt.figure(figsize=(14, 6))
plt.plot(range(1, 101, 5), pom_time, color='c', label="pomegranate")
plt.plot(range(1, 101, 5), skl_time, color='m', label="sklearn")
plt.xticks(fontsize=14)
plt.xlabel("Number of Dimensions", fontsize=14)
plt.yticks(fontsize=14)
plt.ylabel("Time (s)")
plt.legend(fontsize=14)
plt.show()

tilt_a = [[-2, 0.5], [5, 2]]
tilt_b = [[-1, 1.5], [3, 3]]

X = numpy.concatenate((numpy.random.normal(4, 1, size=(250, 2)).dot(tilt_a), numpy.random.normal(3, 1, size=(800, 2)).dot(tilt_b)))
y = numpy.concatenate((numpy.zeros(250), numpy.ones(800)))

model_a = NaiveBayes.from_samples(NormalDistribution, X, y)
model_b = BayesClassifier.from_samples(MultivariateGaussianDistribution, X, y)

xx, yy = np.meshgrid(np.arange(-5, 30, 0.02), np.arange(0, 25, 0.02))
Z1 = model_a.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
Z2 = model_b.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.figure(figsize=(18, 8))
plt.subplot(121)
plt.contour(xx, yy, Z1)
plt.scatter(X[y == 0, 0], X[y == 0, 1], color='c', alpha=0.3)
plt.scatter(X[y == 1, 0], X[y == 1, 1], color='m', alpha=0.3)
plt.xlim(-5, 30)
plt.ylim(0, 25)

plt.subplot(122)
plt.contour(xx, yy, Z2)
plt.scatter(X[y == 0, 0], X[y == 0, 1], color='c', alpha=0.3)
plt.scatter(X[y == 1, 0], X[y == 1, 1], color='m', alpha=0.3)
plt.xlim(-5, 30)
plt.ylim(0, 25)
plt.show()

print "naive training accuracy: {:4.4}".format((model_a.predict(X) == y).mean())
print "bayes classifier training accuracy: {:4.4}".format((model_b.predict(X) == y).mean())

X = numpy.empty(shape=(0, 2))
X = numpy.concatenate((X, numpy.random.normal(4, 1, size=(200, 2)).dot([[-2, 0.5], [2, 0.5]])))
X = numpy.concatenate((X, numpy.random.normal(3, 1, size=(350, 2)).dot([[-1, 2], [1, 0.8]])))
X = numpy.concatenate((X, numpy.random.normal(7, 1, size=(700, 2)).dot([[-0.75, 0.8], [0.9, 1.5]])))
X = numpy.concatenate((X, numpy.random.normal(6, 1, size=(120, 2)).dot([[-1.5, 1.2], [0.6, 1.2]])))
y = numpy.concatenate((numpy.zeros(550), numpy.ones(820)))

model_a = BayesClassifier.from_samples(MultivariateGaussianDistribution, X, y)

gmm_a = GeneralMixtureModel.from_samples(MultivariateGaussianDistribution, 2, X[y == 0])
gmm_b = GeneralMixtureModel.from_samples(MultivariateGaussianDistribution, 2, X[y == 1])
model_b = BayesClassifier([gmm_a, gmm_b], weights=numpy.array([1-y.mean(), y.mean()]))

xx, yy = np.meshgrid(np.arange(-10, 10, 0.02), np.arange(0, 25, 0.02))
Z1 = model_a.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
Z2 = model_b.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
centroids1 = numpy.array([distribution.mu for distribution in model_a.distributions])
centroids2 = numpy.concatenate([[distribution.mu for distribution in component.distributions] for component in model_b.distributions])

plt.figure(figsize=(18, 8))
plt.subplot(121)
plt.contour(xx, yy, Z1)
plt.scatter(X[y == 0, 0], X[y == 0, 1], color='c', alpha=0.3)
plt.scatter(X[y == 1, 0], X[y == 1, 1], color='m', alpha=0.3)
plt.scatter(centroids1[:,0], centroids1[:,1], color='k', s=100)

plt.subplot(122)
plt.contour(xx, yy, Z2)
plt.scatter(X[y == 0, 0], X[y == 0, 1], color='c', alpha=0.3)
plt.scatter(X[y == 1, 0], X[y == 1, 1], color='m', alpha=0.3)
plt.scatter(centroids2[:,0], centroids2[:,1], color='k', s=100)
plt.show()

