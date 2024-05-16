def func1(a, b):
    return a / b

def func2(x):
    a = x
    b = x - 1
    return func1(a, b)

func2(1)

get_ipython().magic('xmode Plain')

func2(1)

get_ipython().magic('xmode Verbose')

func2(1)

get_ipython().magic('debug')

get_ipython().magic('debug')

get_ipython().magic('xmode Plain')
get_ipython().magic('pdb on')
func2(1)

