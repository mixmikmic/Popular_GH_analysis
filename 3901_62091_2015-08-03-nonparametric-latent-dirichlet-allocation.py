get_ipython().magic('matplotlib inline')
get_ipython().magic('precision 2')

vocabulary = ['see', 'spot', 'run']
num_terms = len(vocabulary)
num_topics = 2 # K
num_documents = 5 # M
mean_document_length = 5 # xi
term_dirichlet_parameter = 1 # beta
topic_dirichlet_parameter = 1 # alpha

from scipy.stats import dirichlet, poisson
from numpy import round
from collections import defaultdict
from random import choice as stl_choice

term_dirichlet_vector = num_terms * [term_dirichlet_parameter]
term_distributions = dirichlet(term_dirichlet_vector, 2).rvs(size=num_topics)
print term_distributions

base_distribution = lambda: stl_choice(term_distributions)
# A sample from base_distribution is a distribution over terms
# Each of our two topics has equal probability
from collections import Counter
for topic, count in Counter([tuple(base_distribution()) for _ in range(10000)]).most_common():
    print "count:", count, "topic:", [round(prob, 2) for prob in topic]

from scipy.stats import beta
from numpy.random import choice

class DirichletProcessSample():
    def __init__(self, base_measure, alpha):
        self.base_measure = base_measure
        self.alpha = alpha
        
        self.cache = []
        self.weights = []
        self.total_stick_used = 0.

    def __call__(self):
        remaining = 1.0 - self.total_stick_used
        i = DirichletProcessSample.roll_die(self.weights + [remaining])
        if i is not None and i < len(self.weights) :
            return self.cache[i]
        else:
            stick_piece = beta(1, self.alpha).rvs() * remaining
            self.total_stick_used += stick_piece
            self.weights.append(stick_piece)
            new_value = self.base_measure()
            self.cache.append(new_value)
            return new_value
      
    @staticmethod 
    def roll_die(weights):
        if weights:
            return choice(range(len(weights)), p=weights)
        else:
            return None

topic_distribution = DirichletProcessSample(base_measure=base_distribution, 
                                            alpha=topic_dirichlet_parameter)

for topic, count in Counter([tuple(topic_distribution()) for _ in range(10000)]).most_common():
    print "count:", count, "topic:", [round(prob, 2) for prob in topic]

topic_index = defaultdict(list)
documents = defaultdict(list)

for doc in range(num_documents):
    topic_distribution_rvs = DirichletProcessSample(base_measure=base_distribution, 
                                                    alpha=topic_dirichlet_parameter)
    document_length = poisson(mean_document_length).rvs()
    for word in range(document_length):
        topic_distribution = topic_distribution_rvs()
        topic_index[doc].append(tuple(topic_distribution))
        documents[doc].append(choice(vocabulary, p=topic_distribution))

for doc in documents.values():
    print doc

for i, doc in enumerate(Counter(term_dist).most_common() for term_dist in topic_index.values()):
    print "Doc:", i
    for topic, count in doc:
        print  5*" ", "count:", count, "topic:", [round(prob, 2) for prob in topic]

term_dirichlet_vector = num_terms * [term_dirichlet_parameter]
base_distribution = lambda: dirichlet(term_dirichlet_vector).rvs(size=1)[0]

base_dp_parameter = 10
base_dp = DirichletProcessSample(base_distribution, alpha=base_dp_parameter)

nested_dp_parameter = 10

topic_index = defaultdict(list)
documents = defaultdict(list)

for doc in range(num_documents):
    topic_distribution_rvs = DirichletProcessSample(base_measure=base_dp, 
                                                    alpha=nested_dp_parameter)
    document_length = poisson(mean_document_length).rvs()
    for word in range(document_length):
        topic_distribution = topic_distribution_rvs()
        topic_index[doc].append(tuple(topic_distribution))
        documents[doc].append(choice(vocabulary, p=topic_distribution))

for doc in documents.values():
    print doc

for i, doc in enumerate(Counter(term_dist).most_common() for term_dist in topic_index.values()):
    print "Doc:", i
    for topic, count in doc:
        print  5*" ", "count:", count, "topic:", [round(prob, 2) for prob in topic]

