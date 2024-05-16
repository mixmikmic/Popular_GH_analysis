import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

get_ipython().magic('matplotlib inline')
plt.style.use('ggplot')

proj = pd.read_excel('https://github.com/chris1610/pbpython/blob/master/data/March-2017-forecast-article.xlsx?raw=True')

proj[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head()

proj["Projected_Sessions"] = np.exp(proj.yhat).round()
proj["Projected_Sessions_lower"] = np.exp(proj.yhat_lower).round()
proj["Projected_Sessions_upper"] = np.exp(proj.yhat_upper).round()

final_proj = proj[(proj.ds > "3-5-2017") & 
                  (proj.ds < "5-20-2017")][["ds", "Projected_Sessions_lower", 
                                            "Projected_Sessions", "Projected_Sessions_upper"]]

actual = pd.read_excel('Traffic_20170306-20170519.xlsx')
actual.columns = ["ds", "Actual_Sessions"]

actual.head()

df = pd.merge(actual, final_proj)
df.head()

df["Session_Delta"] = df.Actual_Sessions - df.Projected_Sessions
df.Session_Delta.describe()

# Need to convert to just a date in order to keep plot from throwing errors
df['ds'] = df['ds'].dt.date

fig, ax = plt.subplots(figsize=(9, 6))
df.plot("ds", "Session_Delta", ax=ax)
fig.autofmt_xdate(bottom=0.2, rotation=30, ha='right');

fig, ax = plt.subplots(figsize=(9, 6))
df.plot(kind='line', x='ds', y=['Actual_Sessions', 'Projected_Sessions'], ax=ax, style=['-','--'])
ax.fill_between(df['ds'].values, df['Projected_Sessions_lower'], df['Projected_Sessions_upper'], alpha=0.2)
ax.set(title='Pbpython Traffic Prediction Accuracy', xlabel='', ylabel='Sessions')
fig.autofmt_xdate(bottom=0.2, rotation=30, ha='right')

