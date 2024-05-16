from rl import *

get_ipython().magic('psource PassiveTDAgent')

from mdp import sequential_decision_environment

# Action Directions
north = (0, 1)
south = (0,-1)
west = (-1, 0)
east = (1, 0)

policy = {
    (0, 2): east,  (1, 2): east,  (2, 2): east,   (3, 2): None,
    (0, 1): north,                (2, 1): north,  (3, 1): None,
    (0, 0): north, (1, 0): west,  (2, 0): west,   (3, 0): west, 
}

our_agent = PassiveTDAgent(policy, sequential_decision_environment, alpha=lambda n: 60./(59+n))

from mdp import value_iteration

print(value_iteration(sequential_decision_environment))

for i in range(200):
    run_single_trial(our_agent,sequential_decision_environment)
print(our_agent.U)

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt

def graph_utility_estimates(agent_program, mdp, no_of_iterations, states_to_graph):
    graphs = {state:[] for state in states_to_graph}
    for iteration in range(1,no_of_iterations+1):
        run_single_trial(agent_program, mdp)
        for state in states_to_graph:
            graphs[state].append((iteration, agent_program.U[state]))
    for state, value in graphs.items():
        state_x, state_y = zip(*value)
        plt.plot(state_x, state_y, label=str(state))
    plt.ylim([0,1.2])
    plt.legend(loc='lower right')
    plt.xlabel('Iterations')
    plt.ylabel('U')

agent = PassiveTDAgent(policy, sequential_decision_environment, alpha=lambda n: 60./(59+n))
graph_utility_estimates(agent, sequential_decision_environment, 500, [(2,2)])

graph_utility_estimates(agent, sequential_decision_environment, 500, [(2,2), (3,2)])

get_ipython().magic('psource QLearningAgent')

q_agent = QLearningAgent(sequential_decision_environment, Ne=5, Rplus=2, 
                         alpha=lambda n: 60./(59+n))

for i in range(200):
    run_single_trial(q_agent,sequential_decision_environment)

q_agent.Q

U = defaultdict(lambda: -1000.) # Very Large Negative Value for Comparison see below.
for state_action, value in q_agent.Q.items():
    state, action = state_action
    if U[state] < value:
                U[state] = value

U

print(value_iteration(sequential_decision_environment))



