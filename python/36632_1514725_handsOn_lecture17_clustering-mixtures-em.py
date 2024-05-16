get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt;
import matplotlib as mpl;
import numpy as np;

def coin_likelihood(roll, bias):
    # P(X | Z, theta)
    numHeads = roll.count("H");
    flips = len(roll);
    return pow(bias, numHeads) * pow(1-bias, flips-numHeads);

def coin_marginal_likelihood(rolls, biasA, biasB):
    # P(X | theta)
    trials = [];
    for roll in rolls:
        h = roll.count("H");
        t = roll.count("T");
        likelihoodA = coin_likelihood(roll, biasA);
        likelihoodB = coin_likelihood(roll, biasB);
        trials.append(np.log(0.5 * (likelihoodA + likelihoodB)));
    return sum(trials);

def plot_coin_likelihood(rolls, thetas=None):
    # grid
    xvals = np.linspace(0.01,0.99,100);
    yvals = np.linspace(0.01,0.99,100);
    X,Y = np.meshgrid(xvals, yvals);
    
    # compute likelihood
    Z = [];
    for i,r in enumerate(X):
        z = []
        for j,c in enumerate(r):
            z.append(coin_marginal_likelihood(rolls,c,Y[i][j]));
        Z.append(z);
    
    # plot
    plt.figure(figsize=(10,8));
    C = plt.contour(X,Y,Z,150);
    cbar = plt.colorbar(C);
    plt.title(r"Likelihood $\log p(\mathcal{X}|\theta_A,\theta_B)$", fontsize=20);
    plt.xlabel(r"$\theta_A$", fontsize=20);
    plt.ylabel(r"$\theta_B$", fontsize=20);
    
    # plot thetas
    if thetas is not None:
        thetas = np.array(thetas);
        plt.plot(thetas[:,0], thetas[:,1], '-k', lw=2.0);
        plt.plot(thetas[:,0], thetas[:,1], 'ok', ms=5.0);

plot_coin_likelihood([ "HTTTHHTHTH", "HHHHTHHHHH", 
         "HTHHHHHTHH", "HTHTTTHHTT", "THHHTHHHTH"]);

def e_step(n_flips, theta_A, theta_B):
    """Produce the expected value for heads_A, tails_A, heads_B, tails_B 
    over n_flipsgiven the coin biases"""
    
    # Replace dummy values with your implementation
    heads_A, tails_A, heads_B, tails_B = n_flips, 0, n_flips, 0
    
    return heads_A, tails_A, heads_B, tails_B

def m_step(heads_A, tails_A, heads_B, tails_B):
    """Produce the values for theta that maximize the expected number of heads/tails"""

    # Replace dummy values with your implementation
    theta_A, theta_B = 0.5, 0.5
    
    return theta_A, theta_B

def coin_em(n_flips, theta_A=None, theta_B=None, maxiter=10):
    # Initial Guess
    theta_A = theta_A or random.random();
    theta_B = theta_B or random.random();
    thetas = [(theta_A, theta_B)];
    # Iterate
    for c in range(maxiter):
        print("#%d:\t%0.2f %0.2f" % (c, theta_A, theta_B));
        heads_A, tails_A, heads_B, tails_B = e_step(n_flips, theta_A, theta_B)
        theta_A, theta_B = m_step(heads_A, tails_A, heads_B, tails_B)
        
    thetas.append((theta_A,theta_B));    
    return thetas, (theta_A,theta_B);

rolls = [ "HTTTHHTHTH", "HHHHTHHHHH", "HTHHHHHTHH", 
          "HTHTTTHHTT", "THHHTHHHTH" ];
thetas, _ = coin_em(rolls, 0.1, 0.3, maxiter=6);

plot_coin_likelihood(rolls, thetas)

import random

def generate_sample(num_flips, prob_choose_a=0.5, a_bias=0.5, b_bias=0.5):
    which_coin = random.random()
    if which_coin < prob_choose_a:
        return "".join(['H' if random.random() < a_bias else 'T' for i in range(num_flips)])
    else:
        return "".join(['H' if random.random() < b_bias else 'T' for i in range(num_flips)])

[generate_sample(10),
 generate_sample(10, prob_choose_a=0.2, a_bias=0.9),
 generate_sample(10, prob_choose_a=0.9, a_bias=0.2, b_bias=0.9)]

flips = [] # your code goes here

# your code goes here

