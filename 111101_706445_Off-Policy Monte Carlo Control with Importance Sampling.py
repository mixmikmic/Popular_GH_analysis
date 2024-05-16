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
from lib.envs.blackjack import BlackjackEnv
matplotlib.style.use('ggplot')

env = BlackjackEnv()

def create_behaviour_policy(nA):
    
    def policy_fn(observation):
        A = np.ones(nA, dtype=float) / nA
        return A
    return policy_fn

def create_target_policy(Q):
    
    def policy_fn(state):
        A = np.zeros_like(Q[state], dtype=float)
        best_action = np.argmax(Q[state])
        A[best_action] = 1.0
        return A
    return policy_fn

def mc_control_off_policy_importance_sampling(behaviour_policy, env, num_episodes, discount=1.0, debug=False):
    
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    C = defaultdict(lambda: np.zeros(env.action_space.n))
    
    target_policy = create_target_policy(Q)
    
    for i_episode in range(1, num_episodes+1):
        
        if debug:
            if i_episode % 100000 == 0:
                print("\rEpisode {}/{}.".format(i_episode, num_episodes))
                
        state = env.reset()
        episode = []
        while(True):
            probs = behaviour_policy(state)
            action = np.random.choice(len(probs), p=probs)
            next_state, reward, done, _ = env.step(action)
            episode.append((state, action, reward))
            if done:
                break
            state = next_state
            
        G = 0.0
        W = 1.0
        
        for t in range(len(episode))[::-1]:
            state, action, reward = episode[t]
            G = discount*G + reward
            C[state][action] += W
            Q[state][action] += (W/C[state][action]) * (G - Q[state][action])
            
            if action != np.argmax(target_policy(state)):
                break
                
            W = W * (target_policy(state)[action]/behaviour_policy(state)[action])
            
    return Q, target_policy

behaviour_policy = create_behaviour_policy(env.action_space.n)
optimal_Q, optimal_policy = mc_control_off_policy_importance_sampling(behaviour_policy, env, num_episodes=500000, debug=True)

V = defaultdict(float)
for state, action_values in optimal_Q.items():
    action_value = np.max(action_values)
    V[state] = action_value
plotting.plot_value_function(V, title="Optimal Value Function")



