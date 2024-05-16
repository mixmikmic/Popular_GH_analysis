get_ipython().magic('matplotlib inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from collections import defaultdict

import sys
if "../" not in sys.path:
  sys.path.append("../")

import gym
import plotting
from lib.envs.gridworld import GridworldEnv
import itertools
matplotlib.style.use('ggplot')

env = GridworldEnv()

env.render()

#Create an initial epsilon soft policy
def epsilon_greedy_policy(Q, epsilon, state, nA):
    A = np.ones(nA, dtype=float) * epsilon / nA
    best_action = np.argmax(Q[state])
    A[best_action] += (1.0 - epsilon)
    return A

def online_sarsa_lambda(env, num_episodes, discount=1.0, epsilon=0.1, alpha=0.5, lbda=0.9, debug=False):
    
    Q = defaultdict(lambda: np.zeros(env.action_space.n, dtype=float))
    
    for i_episode in range(1, num_episodes+1):
        
        if debug:
            if i_episode % 1000 == 0:
                print("\rEpisode {}/{}.".format(i_episode, num_episodes))
                
        E = {key:np.zeros(4, dtype=int) for key in np.arange(16)}
        state = env.reset()
        action_probs = epsilon_greedy_policy(Q, epsilon, state, env.nA)
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
        for t in itertools.count():
            next_state, reward, end, _ = env.step(action)
            
            next_action_probs = epsilon_greedy_policy(Q, epsilon, next_state, env.nA)
            next_action = np.random.choice(np.arange(len(next_action_probs)), p=next_action_probs)
            
            delta = reward + discount*Q[next_state][next_action] - Q[state][action]
            E[state][action] += 1
            
            for s in E.keys():
                for a in E[s]:
                    Q[s][a] += alpha*delta*E[s][a]
                    E[s][a] = discount*lbda*E[s][a]
                    
            if end:
                break
                
            state = next_state
            action = next_action
            
    return Q    

Q = online_sarsa_lambda(env, num_episodes=10000, debug=True)

Q

state = env.reset()
print env.render()
print '#################################'
while(True):
    action = np.argmax(Q[state])
    next_state, reward, done, _ = env.step(action)
    state = next_state
    
    print env.render()
    print '#################################'
    
    if done:
        break
    



