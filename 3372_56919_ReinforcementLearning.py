import numpy as np
from scipy import stats

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context("poster")
sns.set_style("ticks")

# For replication
np.random.seed(1337)

n = 10
bandit_payoff_probs = np.random.rand(n)
print bandit_payoff_probs
print "Bandit with best payoff: %s" % np.argmax(bandit_payoff_probs)

def rewards(p, max_cost=5):
    # Return the total reward equal to the times random number < p
    return np.sum(np.random.rand(max_cost) < p)
    
rewards(0.3)

fig, ax = plt.subplots(3,3, figsize=(20,20))

for i, (axi, p) in enumerate(zip(ax.flatten(), [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])):
    axi.hist([rewards(p) for k in xrange(10000)], bins=range(6))
    axi.set_title("p=%s" % p)

def best_arm(mean_rewards):
    return np.argmax(mean_rewards)
    
best_arm([0.1,0.5, 0.3])

eps = 0.05 # epsilon for randomly using a bandit
num_plays = 500
running_mean_reward = 0
mean_rewards = np.zeros(n)
count_arms = np.zeros(n)
print bandit_payoff_probs

plt.clf()
fig, ax = plt.subplots(3,3, figsize=(30,30))
for i, eps in enumerate([0.05, 0.1, 0.2]):
    mean_rewards = np.zeros(n)
    count_arms = np.zeros(n)
    ax[i,0].set_xlabel("Plays")
    ax[i,0].set_ylabel("Mean Reward (eps = %s)" % eps)
    for j in xrange(1,num_plays+1):
        if np.random.rand() > eps:
            choice = best_arm(mean_rewards)
        else:
            choice = np.random.randint(n)
        curr_reward = rewards(bandit_payoff_probs[choice])
        count_arms[choice] += 1
        mean_rewards[choice] += (curr_reward - mean_rewards[choice]) * 1. / count_arms[choice]
        running_mean_reward += (curr_reward - running_mean_reward) * 1. / j
        ax[i,0].scatter(j,running_mean_reward)

    width = 0.4
    ax[i,1].bar(np.arange(n), count_arms * 1. / num_plays, width, color="r", label="Selected")
    ax[i,1].bar(np.arange(n) + width, bandit_payoff_probs, width, color="b", label="Payoff")
    ax[i,1].set_xlabel("Bandit")
    ax[i,1].set_ylabel("p(selected) and p(payoff)")
    ax[i,1].legend(loc="upper right")
    ax[i,2].bar(np.arange(n), mean_rewards)
    ax[i,2].set_xlabel("Bandit")
    ax[i,2].set_ylabel("Mean Reward")


def best_arm(mean_rewards, tau=1.0):
    exp_r = np.exp(mean_rewards/tau)
    exp_r = exp_r / exp_r.sum()
    return np.random.choice(range(n), p=exp_r, size=1)[0]

[best_arm(mean_rewards) for k in xrange(10)]

plt.hist(np.random.choice([1,2,3], p=[0.5,0.1,0.4], size=100))

num_plays = 500
running_mean_reward = 0

plt.clf()
fig, ax = plt.subplots(3,3, figsize=(30,30))
for i, tau in enumerate([0.9, 1., 1.1]):
    mean_rewards = np.zeros(n)
    count_arms = np.zeros(n)
    ax[i,0].set_xlabel("Plays")
    ax[i,0].set_ylabel("Mean Reward (tau = %s)" % tau)
    for j in xrange(1,num_plays+1):
        choice = best_arm(mean_rewards, tau=tau)
        curr_reward = rewards(bandit_payoff_probs[choice])
        count_arms[choice] += 1
        mean_rewards[choice] += (curr_reward - mean_rewards[choice]) * 1. / count_arms[choice]
        running_mean_reward += (curr_reward - running_mean_reward) * 1. / j
        ax[i,0].scatter(j,running_mean_reward)

    width = 0.4
    ax[i,1].bar(np.arange(n), count_arms * 1. / num_plays, width, color="r", label="Selected")
    ax[i,1].bar(np.arange(n) + width, bandit_payoff_probs, width, color="b", label="Payoff")
    ax[i,1].set_xlabel("Bandit")
    ax[i,1].set_ylabel("p(selected) and p(payoff)")
    ax[i,1].legend(loc="upper right")
    ax[i,2].bar(np.arange(n), mean_rewards)
    ax[i,2].set_xlabel("Bandit")
    ax[i,2].set_ylabel("Mean Reward")




