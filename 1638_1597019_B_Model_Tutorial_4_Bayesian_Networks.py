get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn; seaborn.set_style('whitegrid')
import numpy

from pomegranate import *

numpy.random.seed(0)
numpy.set_printoptions(suppress=True)

get_ipython().run_line_magic('load_ext', 'watermark')
get_ipython().run_line_magic('watermark', '-m -n -p numpy,scipy,pomegranate')

# The guests initial door selection is completely random
guest = DiscreteDistribution({'A': 1./3, 'B': 1./3, 'C': 1./3})

# The door the prize is behind is also completely random
prize = DiscreteDistribution({'A': 1./3, 'B': 1./3, 'C': 1./3})

    # Monty is dependent on both the guest and the prize. 
monty = ConditionalProbabilityTable(
        [[ 'A', 'A', 'A', 0.0 ],
         [ 'A', 'A', 'B', 0.5 ],
         [ 'A', 'A', 'C', 0.5 ],
         [ 'A', 'B', 'A', 0.0 ],
         [ 'A', 'B', 'B', 0.0 ],
         [ 'A', 'B', 'C', 1.0 ],
         [ 'A', 'C', 'A', 0.0 ],
         [ 'A', 'C', 'B', 1.0 ],
         [ 'A', 'C', 'C', 0.0 ],
         [ 'B', 'A', 'A', 0.0 ],
         [ 'B', 'A', 'B', 0.0 ],
         [ 'B', 'A', 'C', 1.0 ],
         [ 'B', 'B', 'A', 0.5 ],
         [ 'B', 'B', 'B', 0.0 ],
         [ 'B', 'B', 'C', 0.5 ],
         [ 'B', 'C', 'A', 1.0 ],
         [ 'B', 'C', 'B', 0.0 ],
         [ 'B', 'C', 'C', 0.0 ],
         [ 'C', 'A', 'A', 0.0 ],
         [ 'C', 'A', 'B', 1.0 ],
         [ 'C', 'A', 'C', 0.0 ],
         [ 'C', 'B', 'A', 1.0 ],
         [ 'C', 'B', 'B', 0.0 ],
         [ 'C', 'B', 'C', 0.0 ],
         [ 'C', 'C', 'A', 0.5 ],
         [ 'C', 'C', 'B', 0.5 ],
         [ 'C', 'C', 'C', 0.0 ]], [guest, prize])  

# State objects hold both the distribution, and a high level name.
s1 = State(guest, name="guest")
s2 = State(prize, name="prize")
s3 = State(monty, name="monty")

# Create the Bayesian network object with a useful name
model = BayesianNetwork("Monty Hall Problem")

# Add the three states to the network 
model.add_states(s1, s2, s3)

# Add edges which represent conditional dependencies, where the second node is 
# conditionally dependent on the first node (Monty is dependent on both guest and prize)
model.add_edge(s1, s3)
model.add_edge(s2, s3)

model.bake()

model.probability(['A', 'B', 'C'])

model.probability(['A', 'B', 'B'])

model.predict_proba({})

model.predict_proba([None, None, None])

model.predict_proba(['A', None, None])

model.predict_proba({'guest': 'A', 'monty': 'C'})

from sklearn.datasets import load_digits

data = load_digits()
X, _ = data.data, data.target

plt.imshow(X[0].reshape(8, 8))
plt.grid(False)
plt.show()

X = X[:,:16]
X = (X > numpy.median(X)).astype('float64')

numpy.random.seed(111)

i = numpy.random.randint(X.shape[0], size=10000)
j = numpy.random.randint(X.shape[1], size=10000)

X_missing = X.copy()
X_missing[i, j] = numpy.nan
X_missing

from fancyimpute import SimpleFill

y_pred = SimpleFill().fit_transform(X_missing)[i, j]
numpy.abs(y_pred - X[i, j]).mean()

from fancyimpute import IterativeSVD

y_pred = IterativeSVD(verbose=False).fit_transform(X_missing)[i, j]
numpy.abs(y_pred - X[i, j]).mean()

y_hat = BayesianNetwork.from_samples(X_missing, max_parents=1).predict(X_missing)
numpy.abs(numpy.array(y_hat)[i, j] - X[i, j]).mean()

from fancyimpute import KNN

y_pred = KNN(verbose=False).fit_transform(X_missing)[i, j]
numpy.abs(y_pred - X[i, j]).mean()

d1 = DiscreteDistribution({True: 0.2, False: 0.8})
d2 = DiscreteDistribution({True: 0.6, False: 0.4})
d3 = ConditionalProbabilityTable(
        [[True,  True,  True,  0.2],
         [True,  True,  False, 0.8],
         [True,  False, True,  0.3],
         [True,  False, False, 0.7],
         [False, True,  True,  0.9],
         [False, True,  False, 0.1],
         [False, False, True,  0.4],
         [False, False, False, 0.6]], [d1, d2])

s1 = State(d1, name="s1")
s2 = State(d2, name="s2")
s3 = State(d3, name="s3")

model = BayesianNetwork()
model.add_states(s1, s2, s3)
model.add_edge(s1, s3)
model.add_edge(s2, s3)
model.bake()

numpy.random.seed(111)

X = numpy.random.randint(2, size=(15, 15))
X[:,5] = X[:,4] = X[:,3]
X[:,11] = X[:,12] = X[:,13]

model = BayesianNetwork.from_samples(X)
model.plot()

model.structure

model.predict([[False, False, False, False, None, None, False, None, False, None, True, None, None, True, False]])

model.predict_proba([[False, False, False, False, None, None, False, None, False, None, 
                      True, None, None, True, False]])

