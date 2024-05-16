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

from pomegranate import *
get_ipython().run_line_magic('pylab', 'inline')

d1 = DiscreteDistribution({'A': 0.10, 'C': 0.40, 'G': 0.40, 'T': 0.10})
d2 = ConditionalProbabilityTable([['A', 'A', 0.10],
                                ['A', 'C', 0.50],
                                ['A', 'G', 0.30],
                                ['A', 'T', 0.10],
                                ['C', 'A', 0.10],
                                ['C', 'C', 0.40],
                                ['C', 'T', 0.40],
                                ['C', 'G', 0.10],
                                ['G', 'A', 0.05],
                                ['G', 'C', 0.45],
                                ['G', 'G', 0.45],
                                ['G', 'T', 0.05],
                                ['T', 'A', 0.20],
                                ['T', 'C', 0.30],
                                ['T', 'G', 0.30],
                                ['T', 'T', 0.20]], [d1])

clf = MarkovChain([d1, d2])

clf.log_probability( list('CAGCATCAGT') ) 

clf.log_probability( list('C') )

clf.log_probability( list('CACATCACGACTAATGATAAT') )

clf.fit( map( list, ('CAGCATCAGT', 'C', 'ATATAGAGATAAGCT', 'GCGCAAGT', 'GCATTGC', 'CACATCACGACTAATGATAAT') ) )
print clf.log_probability( list('CAGCATCAGT') ) 
print clf.log_probability( list('C') )
print clf.log_probability( list('CACATCACGACTAATGATAAT') )

print clf.distributions[0] 

print clf.distributions[1]

