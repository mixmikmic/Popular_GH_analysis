import pandas as pd

get_ipython().run_line_magic('matplotlib', 'inline')

import glob

dfs = [pd.read_csv(file_name, parse_dates=['creationdate']) 
           for file_name in glob.glob('../data/stackoverflow/*.csv')]
df = pd.concat(dfs)

df.head()

df.groupby([pd.Grouper(key='creationdate', freq='QS'), 'tagname'])   .size()   .unstack('tagname', fill_value=0)   .pipe(lambda x: x.div(x.sum(1), axis=0))   .plot(kind='area', figsize=(12,6), title='Percentage of New Stack Overflow Questions')   .legend(loc='upper left')



