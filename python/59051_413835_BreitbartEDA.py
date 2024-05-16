get_ipython().magic('matplotlib inline')
import pandas as pd
import datetime
import ast
import tldextract

# You will need access to D4D data.world organization. Check in slack for more info
# 150mb / 240k rows. Give it time, has to download over internet
df = pd.read_csv('https://query.data.world/s/bbokc1f08to11j19j5axvkrcv', sep='\t', parse_dates=['date'])

df.set_index('date', inplace=True)
df.count()

by_year=df.groupby([pd.TimeGrouper('A')]).count()['title']
by_year

by_year.plot()

df.groupby([pd.TimeGrouper('A'),'category']).count()['title']

df.groupby(['author']).count()['title'].sort_values(ascending=0).head(25)

from collections import Counter
tld_counter = Counter()

def get_tld(hrefs):
    
    # Quick and dirty, not thorough yet
    for link in ast.literal_eval(hrefs):
        top_level = tldextract.extract(link)
        top_level = top_level.domain
        tld_counter[top_level] += 1

_ = df[['hrefs']].applymap(get_tld)

tld_counter.most_common(25)

