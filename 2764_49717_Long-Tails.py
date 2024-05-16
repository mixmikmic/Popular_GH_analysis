import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().magic('matplotlib inline')
sns.set_context('talk')
sns.set_style('darkgrid')

inventory = 100.0
volume = 5000.0

rr = np.linspace(1,inventory,100)
ns = [0.25, 0.75, 1.25, 1.75]

fig, ax = plt.subplots(figsize=(10, 6))

for nn in ns:
    norm = (nn-1)*volume/(1-inventory**(1-nn))
    ax.plot(rr, norm/rr**nn, label='$n=%g$' % nn)

ax.legend()
ax.set_xlabel('Rank by Sales Volume $r$')
ax.set_ylabel('Units Sold')
ax.set_title('Sales volume of each product by rank')
ax.set_ylim(0,100)

# Same plot as above
fig, ax = plt.subplots(figsize=(10, 6))

for nn in ns:
    norm = (nn-1)*volume/(1-inventory**(1-nn))
    ax.plot(rr, norm/rr**nn, label='$n=%g$' % nn)

ax.set_xlabel('Rank by Sales Volume $r$')
ax.set_ylabel('Units Sold')
ax.set_title('Sales volume of each product by rank')
ax.set_ylim(0,100)

# Ask seaborn for some pleasing colors
c1, c2, c3 = sns.color_palette(n_colors=3)

# Add transparent rectangles
head_patch = plt.matplotlib.patches.Rectangle((1,0), 9, 100, alpha=0.25, color=c1)
middle_patch = plt.matplotlib.patches.Rectangle((11,0), 39, 100, alpha=0.25, color=c2)
tail_patch = plt.matplotlib.patches.Rectangle((51,0), 48, 100, alpha=0.25, color=c3)
ax.add_patch(head_patch)
ax.add_patch(middle_patch)
ax.add_patch(tail_patch)

# Add text annotations
ax.text(5,50,"Head", color=c1, fontsize=16, rotation=90)
ax.text(25,80,"Middle", color=c2, fontsize=16)
ax.text(75,80,"Tail", color=c3, fontsize=16)

f_head = 0.1
f_tail = 0.5

ns = np.linspace(0,2,100)
nm1 = ns-1.0

head = volume*(inventory**nm1 - f_head**-nm1)/(inventory**nm1-1)
middle = volume*(f_head**-nm1 - f_tail**-nm1)/(inventory**nm1-1)
tail = volume*(f_tail**-nm1 - 1)/(inventory**nm1-1)

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(ns, head, label='Head')
ax.plot(ns, middle, label='Middle')
ax.plot(ns, tail, label='Tail')
ax.legend(loc='upper left')
ax.set_ylabel('Units Sold')
ax.set_xlabel('Power law index $n$')

marginal_benefit = ((ns-1)*volume)/((1-inventory**(1-ns))*(inventory+1)**ns)

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(ns, marginal_benefit)
ax.set_ylabel('Additional Units Sold')
ax.set_xlabel('Power law index $n$')
ax.set_title('Marginal Benefit of Expanding Inventory')

