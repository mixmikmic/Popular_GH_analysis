get_ipython().run_line_magic('pylab', 'inline')
import seaborn, time
seaborn.set_style('whitegrid')

from pomegranate import BayesianNetwork
from libpgm.pgmlearner import PGMLearner

libpgm_time = []
pomegranate_time = []
pomegranate_cl_time = []

for i in range(2, 15):
    tic = time.time()
    X = numpy.random.randint(2, size=(10000, i))
    model = BayesianNetwork.from_samples(X, algorithm='exact')
    pomegranate_time.append(time.time() - tic)

    tic = time.time()
    model = BayesianNetwork.from_samples(X, algorithm='chow-liu')
    pomegranate_cl_time.append(time.time() - tic)

    X = [{j : X[i, j] for j in range(X.shape[1])} for i in range(X.shape[0])]
    learner = PGMLearner()

    tic = time.time()
    model = learner.discrete_constraint_estimatestruct(X)
    libpgm_time.append(time.time() - tic)

plt.figure(figsize=(14, 6))
plt.title("Bayesian Network Structure Learning Time", fontsize=16)
plt.xlabel("Number of Variables", fontsize=14)
plt.ylabel("Time (s)", fontsize=14)
plt.plot(range(2, 15), libpgm_time, c='c', label="libpgm")
plt.plot(range(2, 15), pomegranate_time, c='m', label="pomegranate exact")
plt.plot(range(2, 15), pomegranate_cl_time, c='r', label="pomegranate chow liu")
plt.legend(loc=2, fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()

libpgm_time = []
pomegranate_time = []
pomegranate_cl_time = []

x = 10, 25, 100, 250, 1000, 2500, 10000, 25000, 100000, 250000, 1000000
for i in x:
    tic = time.time()
    X = numpy.random.randint(2, size=(i, 10))
    model = BayesianNetwork.from_samples(X, algorithm='exact')
    pomegranate_time.append(time.time() - tic)

    tic = time.time()
    model = BayesianNetwork.from_samples(X, algorithm='chow-liu')
    pomegranate_cl_time.append(time.time() - tic)

    X = [{j : X[i, j] for j in range(X.shape[1])} for i in range(X.shape[0])]
    learner = PGMLearner()

    tic = time.time()
    model = learner.discrete_constraint_estimatestruct(X)
    libpgm_time.append(time.time() - tic)

plt.figure(figsize=(14, 6))
plt.title("Bayesian Network Structure Learning Time", fontsize=16)
plt.xlabel("Number of Samples", fontsize=14)
plt.ylabel("Time (s)", fontsize=14)
plt.plot(x, libpgm_time, c='c', label="libpgm")
plt.plot(x, pomegranate_time, c='m', label="pomegranate exact")
plt.plot(x, pomegranate_cl_time, c='r', label="pomegranate chow liu")
plt.legend(loc=2, fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()

