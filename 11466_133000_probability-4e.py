from collections import defaultdict, Counter
import itertools
import math
import random

class BayesNet(object):
    "Bayesian network: a graph of variables connected by parent links."
     
    def __init__(self): 
        self.variables = [] # List of variables, in parent-first topological sort order
        self.lookup = {}    # Mapping of {variable_name: variable} pairs
            
    def add(self, name, parentnames, cpt):
        "Add a new Variable to the BayesNet. Parentnames must have been added previously."
        parents = [self.lookup[name] for name in parentnames]
        var = Variable(name, cpt, parents)
        self.variables.append(var)
        self.lookup[name] = var
        return self
    
class Variable(object):
    "A discrete random variable; conditional on zero or more parent Variables."
    
    def __init__(self, name, cpt, parents=()):
        "A variable has a name, list of parent variables, and a Conditional Probability Table."
        self.__name__ = name
        self.parents  = parents
        self.cpt      = CPTable(cpt, parents)
        self.domain   = set(itertools.chain(*self.cpt.values())) # All the outcomes in the CPT
                
    def __repr__(self): return self.__name__
    
class Factor(dict): "An {outcome: frequency} mapping."

class ProbDist(Factor):
    """A Probability Distribution is an {outcome: probability} mapping. 
    The values are normalized to sum to 1.
    ProbDist(0.75) is an abbreviation for ProbDist({T: 0.75, F: 0.25})."""
    def __init__(self, mapping=(), **kwargs):
        if isinstance(mapping, float):
            mapping = {T: mapping, F: 1 - mapping}
        self.update(mapping, **kwargs)
        normalize(self)
        
class Evidence(dict): 
    "A {variable: value} mapping, describing what we know for sure."
        
class CPTable(dict):
    "A mapping of {row: ProbDist, ...} where each row is a tuple of values of the parent variables."
    
    def __init__(self, mapping, parents=()):
        """Provides two shortcuts for writing a Conditional Probability Table. 
        With no parents, CPTable(dist) means CPTable({(): dist}).
        With one parent, CPTable({val: dist,...}) means CPTable({(val,): dist,...})."""
        if len(parents) == 0 and not (isinstance(mapping, dict) and set(mapping.keys()) == {()}):
            mapping = {(): mapping}
        for (row, dist) in mapping.items():
            if len(parents) == 1 and not isinstance(row, tuple): 
                row = (row,)
            self[row] = ProbDist(dist)

class Bool(int):
    "Just like `bool`, except values display as 'T' and 'F' instead of 'True' and 'False'"
    __str__ = __repr__ = lambda self: 'T' if self else 'F'
        
T = Bool(True)
F = Bool(False)

def P(var, evidence={}):
    "The probability distribution for P(variable | evidence), when all parent variables are known (in evidence)."
    row = tuple(evidence[parent] for parent in var.parents)
    return var.cpt[row]

def normalize(dist):
    "Normalize a {key: value} distribution so values sum to 1.0. Mutates dist and returns it."
    total = sum(dist.values())
    for key in dist:
        dist[key] = dist[key] / total
        assert 0 <= dist[key] <= 1, "Probabilities must be between 0 and 1."
    return dist

def sample(probdist):
    "Randomly sample an outcome from a probability distribution."
    r = random.random() # r is a random point in the probability distribution
    c = 0.0             # c is the cumulative probability of outcomes seen so far
    for outcome in probdist:
        c += probdist[outcome]
        if r <= c:
            return outcome
        
def globalize(mapping):
    "Given a {name: value} mapping, export all the names to the `globals()` namespace."
    globals().update(mapping)

# Example random variable: Earthquake:
# An earthquake occurs on 0.002 of days, independent of any other variables.
Earthquake = Variable('Earthquake', 0.002)

# The probability distribution for Earthquake
P(Earthquake)

# Get the probability of a specific outcome by subscripting the probability distribution
P(Earthquake)[T]

# Randomly sample from the distribution:
sample(P(Earthquake))

# Randomly sample 100,000 times, and count up the results:
Counter(sample(P(Earthquake)) for i in range(100000))

# Two equivalent ways of specifying the same Boolean probability distribution:
assert ProbDist(0.75) == ProbDist({T: 0.75, F: 0.25})

# Two equivalent ways of specifying the same non-Boolean probability distribution:
assert ProbDist(win=15, lose=3, tie=2) == ProbDist({'win': 15, 'lose': 3, 'tie': 2})
ProbDist(win=15, lose=3, tie=2)

# The difference between a Factor and a ProbDist--the ProbDist is normalized:
Factor(a=1, b=2, c=3, d=4)

ProbDist(a=1, b=2, c=3, d=4)

alarm_net = (BayesNet()
    .add('Burglary', [], 0.001)
    .add('Earthquake', [], 0.002)
    .add('Alarm', ['Burglary', 'Earthquake'], {(T, T): 0.95, (T, F): 0.94, (F, T): 0.29, (F, F): 0.001})
    .add('JohnCalls', ['Alarm'], {T: 0.90, F: 0.05})
    .add('MaryCalls', ['Alarm'], {T: 0.70, F: 0.01}))  

# Make Burglary, Earthquake, etc. be global variables
globalize(alarm_net.lookup) 
alarm_net.variables

# Probability distribution of a Burglary
P(Burglary)

# Probability of Alarm going off, given a Burglary and not an Earthquake:
P(Alarm, {Burglary: T, Earthquake: F})

# Where that came from: the (T, F) row of Alarm's CPT:
Alarm.cpt

def joint_distribution(net):
    "Given a Bayes net, create the joint distribution over all variables."
    return ProbDist({row: prod(P_xi_given_parents(var, row, net)
                               for var in net.variables)
                     for row in all_rows(net)})

def all_rows(net): return itertools.product(*[var.domain for var in net.variables])

def P_xi_given_parents(var, row, net):
    "The probability that var = xi, given the values in this row."
    dist = P(var, Evidence(zip(net.variables, row)))
    xi = row[net.variables.index(var)]
    return dist[xi]

def prod(numbers):
    "The product of numbers: prod([2, 3, 5]) == 30. Analogous to `sum([2, 3, 5]) == 10`."
    result = 1
    for x in numbers:
        result *= x
    return result

# All rows in the joint distribution (2**5 == 32 rows)
set(all_rows(alarm_net))

# Let's work through just one row of the table:
row = (F, F, F, F, F)

# This is the probability distribution for Alarm
P(Alarm, {Burglary: F, Earthquake: F})

# Here's the probability that Alarm is false, given the parent values in this row:
P_xi_given_parents(Alarm, row, alarm_net)

# The full joint distribution:
joint_distribution(alarm_net)

# Probability that "the alarm has sounded, but neither a burglary nor an earthquake has occurred, 
# and both John and Mary call" (page 514 says it should be 0.000628)

print(alarm_net.variables)
joint_distribution(alarm_net)[F, F, T, T, T]

def enumeration_ask(X, evidence, net):
    "The probability distribution for query variable X in a belief net, given evidence."
    i    = net.variables.index(X) # The index of the query variable X in the row
    dist = defaultdict(float)     # The resulting probability distribution over X
    for (row, p) in joint_distribution(net).items():
        if matches_evidence(row, evidence, net):
            dist[row[i]] += p
    return ProbDist(dist)

def matches_evidence(row, evidence, net):
    "Does the tuple of values for this row agree with the evidence?"
    return all(evidence[v] == row[net.variables.index(v)]
               for v in evidence)

# The probability of a Burgalry, given that John calls but Mary does not: 
enumeration_ask(Burglary, {JohnCalls: F, MaryCalls: T}, alarm_net)

# The probability of an Alarm, given that there is an Earthquake and Mary calls:
enumeration_ask(Alarm, {MaryCalls: T, Earthquake: T}, alarm_net)

# TODO: Copy over and update Variable Elimination algorithm. Also, sampling algorithms.

flu_net = (BayesNet()
           .add('Vaccinated', [], 0.60)
           .add('Flu', ['Vaccinated'], {T: 0.002, F: 0.02})
           .add('Fever', ['Flu'], {T: ProbDist(no=25, mild=25, high=50),
                                   F: ProbDist(no=97, mild=2, high=1)})
           .add('Headache', ['Flu'], {T: 0.5,   F: 0.03}))

globalize(flu_net.lookup)

# If you just have a headache, you probably don't have the Flu.
enumeration_ask(Flu, {Headache: T, Fever: 'no'}, flu_net)

# Even more so if you were vaccinated.
enumeration_ask(Flu, {Headache: T, Fever: 'no', Vaccinated: T}, flu_net)

# But if you were not vaccinated, there is a higher chance you have the flu.
enumeration_ask(Flu, {Headache: T, Fever: 'no', Vaccinated: F}, flu_net)

# And if you have both headache and fever, and were not vaccinated, 
# then the flu is very likely, especially if it is a high fever.
enumeration_ask(Flu, {Headache: T, Fever: 'mild', Vaccinated: F}, flu_net)

enumeration_ask(Flu, {Headache: T, Fever: 'high', Vaccinated: F}, flu_net)

def entropy(probdist):
    "The entropy of a probability distribution."
    return - sum(p * math.log(p, 2)
                 for p in probdist.values())

entropy(ProbDist(heads=0.5, tails=0.5))

entropy(ProbDist(yes=1000, no=1))

entropy(P(Alarm, {Earthquake: T, Burglary: F}))

entropy(P(Alarm, {Earthquake: F, Burglary: F}))

entropy(P(Fever, {Flu: T}))

class DefaultProbDist(ProbDist):
    """A Probability Distribution that supports smoothing for unknown outcomes (keys).
    The default_value represents the probability of an unknown (previously unseen) key. 
    The key `None` stands for unknown outcomes."""
    def __init__(self, default_value, mapping=(), **kwargs):
        self[None] = default_value
        self.update(mapping, **kwargs)
        normalize(self)
        
    def __missing__(self, key): return self[None]        

import re

def words(text): return re.findall(r'\w+', text.lower())

english = words('''This is a sample corpus of English prose. To get a better model, we would train on much
more text. But this should give you an idea of the process. So far we have dealt with discrete 
distributions where we  know all the possible outcomes in advance. For Boolean variables, the only 
outcomes are T and F. For Fever, we modeled exactly three outcomes. However, in some applications we 
will encounter new, previously unknown outcomes over time. For example, when we could train a model on the 
words in this text, we get a distribution, but somebody could coin a brand new word. To deal with this, 
we introduce the DefaultProbDist distribution, which uses the key `None` to stand as a placeholder for any 
unknown outcomes. Probability theory allows us to compute the likelihood of certain events, given 
assumptions about the components of the event. A Bayesian network, or Bayes net for short, is a data 
structure to represent a joint probability distribution over several random variables, and do inference on it.''')

E = DefaultProbDist(0.1, Counter(english))

# 'the' is a common word:
E['the']

# 'possible' is a less-common word:
E['possible']

# 'impossible' was not seen in the training data, but still gets a non-zero probability ...
E['impossible']

# ... as do other rare, previously unseen words:
E['llanfairpwllgwyngyll']

E[None]

