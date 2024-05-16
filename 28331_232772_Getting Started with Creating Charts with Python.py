import swat

conn = swat.CAS(host, port, username, password)

tbl = conn.read_csv('https://raw.githubusercontent.com/sassoftware/sas-viya-programming/master/data/cars.csv')
tbl.head()

sports = tbl.query('Type = "Sports"')
sports

df = sports.head(1000)
df.set_index(df['Make'] + ' ' + df['Model'], inplace=True)
df.head()

get_ipython().magic('matplotlib inline')

df[['MSRP', 'Invoice']].plot.bar(figsize=(15, 8), rot=-90, subplots=True)

import seaborn as sns
import matplotlib.pyplot as plt

f, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), sharex=True)

bar = sns.barplot(df.index, df['MSRP'], ax=ax1, color='blue')
ax1.set_ylabel('MSRP')

bar2 = sns.barplot(df.index, df['Invoice'], ax=ax2, color='green')
ax2.set_ylabel('Invoice')

labels = bar2.set_xticklabels(df.index, rotation=-90)

import matplotlib.pyplot as plt

f, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), sharex=True)

ax1.bar(range(len(df.index)), df['MSRP'], color='blue')
ax1.set_ylabel('MSRP')

ax2.bar(range(len(df.index)), df['Invoice'], color='green')
ax2.set_ylabel('Invoice')

ax2.set_xticks([x + 0.25 for x in range(len(df.index))])
labels = ax2.set_xticklabels(df.index, rotation=-90)

import cufflinks as cf

cf.go_offline()

df[['MSRP', 'Invoice']].iplot(kind='bar', subplots=True, shape=(2, 1), shared_xaxes=True)

import plotly.graph_objs as go
from plotly import tools
from plotly.offline import init_notebook_mode, iplot

init_notebook_mode()

data = [
    go.Bar(x=df.index, y=df.MSRP, name='MSRP'),
    go.Bar(x=df.index, y=df.Invoice, name='Invoice')
]

fig = tools.make_subplots(rows=2, cols=1, shared_xaxes=True, print_grid=True)
fig.append_trace(data[0], 1, 1)
fig.append_trace(data[1], 2, 1)

fig['layout']['height'] = 700
fig['layout']['margin'] = dict(b=250)

iplot(fig)

from bokeh.charts import Bar, show
from bokeh.io import output_notebook

output_notebook()

try: show(Bar(df, values='MSRP', ylabel='MSRP', width=1000, height=400, color='blue'))
except: pass
try: show(Bar(df, values='Invoice', ylabel='Invoice', width=1000, height=400, color='green'))
except: pass

conn.close()



