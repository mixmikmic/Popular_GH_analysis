get_ipython().run_cell_magic('HTML', '', '<iframe width="560" height="315" src="https://www.youtube.com/embed/XRO6lEu9-5w" frameborder="0" allowfullscreen></iframe>')

import pandas.io.data as web

df = web.DataReader('AAPL', 'google', '2016/1/1', '2017/1/1')

df.head()

get_ipython().magic('matplotlib inline')
df.plot(y='Close', color="Green")

df.plot.bar(y='Volume')



get_ipython().magic('pinfo df.plot.bar')

get_ipython().magic('lsmagic')

get_ipython().magic('time for i in range(100000): i*i')

get_ipython().magic('system pwd')

get_ipython().system('ls')

