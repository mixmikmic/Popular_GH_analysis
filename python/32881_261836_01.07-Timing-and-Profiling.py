get_ipython().magic('timeit sum(range(100))')

get_ipython().run_cell_magic('timeit', '', 'total = 0\nfor i in range(1000):\n    for j in range(1000):\n        total += i * (-1) ** j')

import random
L = [random.random() for i in range(100000)]
get_ipython().magic('timeit L.sort()')

import random
L = [random.random() for i in range(100000)]
print("sorting an unsorted list:")
get_ipython().magic('time L.sort()')

print("sorting an already sorted list:")
get_ipython().magic('time L.sort()')

get_ipython().run_cell_magic('time', '', 'total = 0\nfor i in range(1000):\n    for j in range(1000):\n        total += i * (-1) ** j')

def sum_of_lists(N):
    total = 0
    for i in range(5):
        L = [j ^ (j >> i) for j in range(N)]
        total += sum(L)
    return total

get_ipython().magic('prun sum_of_lists(1000000)')

get_ipython().magic('load_ext line_profiler')

get_ipython().magic('lprun -f sum_of_lists sum_of_lists(5000)')

get_ipython().magic('load_ext memory_profiler')

get_ipython().magic('memit sum_of_lists(1000000)')

get_ipython().run_cell_magic('file', 'mprun_demo.py', 'def sum_of_lists(N):\n    total = 0\n    for i in range(5):\n        L = [j ^ (j >> i) for j in range(N)]\n        total += sum(L)\n        del L # remove reference to L\n    return total')

from mprun_demo import sum_of_lists
get_ipython().magic('mprun -f sum_of_lists sum_of_lists(1000000)')

