from probability import *

get_ipython().magic('psource ProbDist')

p = ProbDist('Flip')
p['H'], p['T'] = 0.25, 0.75
p['T']

p = ProbDist(freqs={'low': 125, 'medium': 375, 'high': 500})
p.varname

(p['low'], p['medium'], p['high'])

p.values

p = ProbDist('Y')
p['Cat'] = 50
p['Dog'] = 114
p['Mice'] = 64
(p['Cat'], p['Dog'], p['Mice'])

p.normalize()
(p['Cat'], p['Dog'], p['Mice'])

p.show_approx()

event = {'A': 10, 'B': 9, 'C': 8}
variables = ['C', 'A']
event_values (event, variables)

get_ipython().magic('psource JointProbDist')

variables = ['X', 'Y']
j = JointProbDist(variables)
j

j[1,1] = 0.2
j[dict(X=0, Y=1)] = 0.5

(j[1,1], j[0,1])

j.values('X')

full_joint = JointProbDist(['Cavity', 'Toothache', 'Catch'])
full_joint[dict(Cavity=True, Toothache=True, Catch=True)] = 0.108
full_joint[dict(Cavity=True, Toothache=True, Catch=False)] = 0.012
full_joint[dict(Cavity=True, Toothache=False, Catch=True)] = 0.016
full_joint[dict(Cavity=True, Toothache=False, Catch=False)] = 0.064
full_joint[dict(Cavity=False, Toothache=True, Catch=True)] = 0.072
full_joint[dict(Cavity=False, Toothache=False, Catch=True)] = 0.144
full_joint[dict(Cavity=False, Toothache=True, Catch=False)] = 0.008
full_joint[dict(Cavity=False, Toothache=False, Catch=False)] = 0.576

get_ipython().magic('psource enumerate_joint')

evidence = dict(Toothache=True)
variables = ['Cavity', 'Catch'] # variables not part of evidence
ans1 = enumerate_joint(variables, evidence, full_joint)
ans1

evidence = dict(Cavity=True, Toothache=True)
variables = ['Catch'] # variables not part of evidence
ans2 = enumerate_joint(variables, evidence, full_joint)
ans2

ans2/ans1

get_ipython().magic('psource enumerate_joint_ask')

query_variable = 'Cavity'
evidence = dict(Toothache=True)
ans = enumerate_joint_ask(query_variable, evidence, full_joint)
(ans[True], ans[False])

get_ipython().magic('psource BayesNode')

alarm_node = BayesNode('Alarm', ['Burglary', 'Earthquake'], 
                       {(True, True): 0.95,(True, False): 0.94, (False, True): 0.29, (False, False): 0.001})

john_node = BayesNode('JohnCalls', ['Alarm'], {True: 0.90, False: 0.05})
mary_node = BayesNode('MaryCalls', 'Alarm', {(True, ): 0.70, (False, ): 0.01}) # Using string for parents.
# Equvivalant to john_node definition. 

burglary_node = BayesNode('Burglary', '', 0.001)
earthquake_node = BayesNode('Earthquake', '', 0.002)

john_node.p(False, {'Alarm': True, 'Burglary': True}) # P(JohnCalls=False | Alarm=True)

get_ipython().magic('psource BayesNet')

burglary

type(burglary.variable_node('Alarm'))

burglary.variable_node('Alarm').cpt

get_ipython().magic('psource enumerate_all')

get_ipython().magic('psource enumeration_ask')

ans_dist = enumeration_ask('Burglary', {'JohnCalls': True, 'MaryCalls': True}, burglary)
ans_dist[True]

get_ipython().magic('psource make_factor')

get_ipython().magic('psource all_events')

f5 = make_factor('MaryCalls', {'JohnCalls': True, 'MaryCalls': True}, burglary)

f5

f5.cpt

f5.variables

new_factor = make_factor('MaryCalls', {'Alarm': True}, burglary)

new_factor.cpt

get_ipython().magic('psource Factor.pointwise_product')

get_ipython().magic('psource pointwise_product')

get_ipython().magic('psource Factor.sum_out')

get_ipython().magic('psource sum_out')

get_ipython().magic('psource elimination_ask')

elimination_ask('Burglary', dict(JohnCalls=True, MaryCalls=True), burglary).show_approx()

get_ipython().magic('psource BayesNode.sample')

get_ipython().magic('psource prior_sample')

N = 1000
all_observations = [prior_sample(sprinkler) for x in range(N)]

rain_true = [observation for observation in all_observations if observation['Rain'] == True]

answer = len(rain_true) / N
print(answer)

rain_and_cloudy = [observation for observation in rain_true if observation['Cloudy'] == True]
answer = len(rain_and_cloudy) / len(rain_true)
print(answer)

get_ipython().magic('psource rejection_sampling')

get_ipython().magic('psource consistent_with')

p = rejection_sampling('Cloudy', dict(Rain=True), sprinkler, 1000)
p[True]

get_ipython().magic('psource weighted_sample')

weighted_sample(sprinkler, dict(Rain=True))

get_ipython().magic('psource likelihood_weighting')

likelihood_weighting('Cloudy', dict(Rain=True), sprinkler, 200).show_approx()

get_ipython().magic('psource gibbs_ask')

gibbs_ask('Cloudy', dict(Rain=True), sprinkler, 200).show_approx()

