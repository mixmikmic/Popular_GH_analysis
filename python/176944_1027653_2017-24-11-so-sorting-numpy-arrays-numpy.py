import numpy as np

names = np.array(['F', 'C', 'A', 'G'])
weights = np.array([20.8, 93.2, 53.4, 61.8])

sort(weights)

#argsort
ordered_indices = np.argsort(weights)
ordered_indices

weights[ordered_indices]

names[ordered_indices]

data = np.array([20.8,  93.2,  53.4,  61.8])
data.argsort()

# sort data
data.sort()
data

# 2d array
a = np.array([
        [.2, .1, .5], 
        [.4, .8, .3],
        [.9, .6, .7]
    ])
a

sort(a)

# sort by column
sort(a, axis = 0)

# search sort
sorted_array = linspace(0,1,5)
values = array([.1,.8,.3,.12,.5,.25])

np.searchsorted(sorted_array, values)







