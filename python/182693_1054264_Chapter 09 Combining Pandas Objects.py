import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

names = pd.read_csv('data/names.csv')
names

new_data_list = ['Aria', 1]
names.loc[4] = new_data_list
names

names.loc['five'] = ['Zach', 3]
names

names.loc[len(names)] = {'Name':'Zayd', 'Age':2}
names

names

names.loc[len(names)] = pd.Series({'Age':32, 'Name':'Dean'})
names

# Use append with fresh copy of names
names = pd.read_csv('data/names.csv')
names.append({'Name':'Aria', 'Age':1})

names.append({'Name':'Aria', 'Age':1}, ignore_index=True)

names.index = ['Canada', 'Canada', 'USA', 'USA']
names

names.append({'Name':'Aria', 'Age':1}, ignore_index=True)

s = pd.Series({'Name': 'Zach', 'Age': 3}, name=len(names))
s

names.append(s)

s1 = pd.Series({'Name': 'Zach', 'Age': 3}, name=len(names))
s2 = pd.Series({'Name': 'Zayd', 'Age': 2}, name='USA')
names.append([s1, s2])

bball_16 = pd.read_csv('data/baseball16.csv')
bball_16.head()

data_dict = bball_16.iloc[0].to_dict()
print(data_dict)

new_data_dict = {k: '' if isinstance(v, str) else np.nan for k, v in data_dict.items()}
print(new_data_dict)

random_data = []
for i in range(1000):
    d = dict()
    for k, v in data_dict.items():
        if isinstance(v, str):
            d[k] = np.random.choice(list('abcde'))
        else:
            d[k] = np.random.randint(10)
    random_data.append(pd.Series(d, name=i + len(bball_16)))
    
random_data[0].head()

get_ipython().run_cell_magic('timeit', '', 'bball_16_copy = bball_16.copy()\nfor row in random_data:\n    bball_16_copy = bball_16_copy.append(row)')

get_ipython().run_cell_magic('timeit', '', 'bball_16_copy = bball_16.copy()\nbball_16_copy = bball_16_copy.append(random_data)')

stocks_2016 = pd.read_csv('data/stocks_2016.csv', index_col='Symbol')
stocks_2017 = pd.read_csv('data/stocks_2017.csv', index_col='Symbol')

stocks_2016

stocks_2017

s_list = [stocks_2016, stocks_2017]
pd.concat(s_list)

pd.concat(s_list, keys=['2016', '2017'], names=['Year', 'Symbol'])

pd.concat(s_list, keys=['2016', '2017'], axis='columns', names=['Year', None])

pd.concat(s_list, join='inner', keys=['2016', '2017'], axis='columns', names=['Year', None])

stocks_2016.append(stocks_2017)

stocks_2015 = stocks_2016.copy()

stocks_2017

base_url = 'http://www.presidency.ucsb.edu/data/popularity.php?pres={}'
trump_url = base_url.format(45)

df_list = pd.read_html(trump_url)
len(df_list)

df0 = df_list[0]
df0.shape

df0.head(7)

df_list = pd.read_html(trump_url, match='Start Date')
len(df_list)

df_list = pd.read_html(trump_url, match='Start Date', attrs={'align':'center'})
len(df_list)

trump = df_list[0]
trump.shape

trump.head(8)

df_list = pd.read_html(trump_url, match='Start Date', attrs={'align':'center'}, 
                       header=0, skiprows=[0,1,2,3,5], parse_dates=['Start Date', 'End Date'])
trump = df_list[0]
trump.head()

trump = trump.dropna(axis=1, how='all')
trump.head()

trump.isnull().sum()

trump = trump.ffill()
trump.head()

trump.dtypes

def get_pres_appr(pres_num):
    base_url = 'http://www.presidency.ucsb.edu/data/popularity.php?pres={}'
    pres_url = base_url.format(pres_num)
    df_list = pd.read_html(pres_url, match='Start Date', attrs={'align':'center'}, 
                       header=0, skiprows=[0,1,2,3,5], parse_dates=['Start Date', 'End Date'])
    pres = df_list[0].copy()
    pres = pres.dropna(axis=1, how='all')
    pres['President'] = pres['President'].ffill()
    return pres.sort_values('End Date').reset_index(drop=True)

obama = get_pres_appr(44)
obama.head()

pres_41_45 = pd.concat([get_pres_appr(x) for x in range(41,46)], ignore_index=True)
pres_41_45.groupby('President').head(3)

pres_41_45['End Date'].value_counts().head(8)

pres_41_45 = pres_41_45.drop_duplicates(subset='End Date')

pres_41_45.shape

pres_41_45['President'].value_counts()

pres_41_45.groupby('President', sort=False).median().round(1)

from matplotlib import cm
fig, ax = plt.subplots(figsize=(16,6))

styles = ['-.', '-', ':', '-', ':']
colors = [.9, .3, .7, .3, .9]
groups = pres_41_45.groupby('President', sort=False)

for style, color, (pres, df) in zip(styles, colors, groups):
    df.plot('End Date', 'Approving', ax=ax, label=pres, style=style, color=cm.Greys(color), 
            title='Presedential Approval Rating')

days_func = lambda x: x - x.iloc[0]
pres_41_45['Days in Office'] = pres_41_45.groupby('President')                                              ['End Date']                                              .transform(days_func)

pres_41_45['Days in Office'] = pres_41_45.groupby('President')['End Date'].transform(lambda x: x - x.iloc[0])
pres_41_45.groupby('President').head(3)

pres_41_45.dtypes

pres_41_45['Days in Office'] = pres_41_45['Days in Office'].dt.days
pres_41_45['Days in Office'].head()

pres_pivot = pres_41_45.pivot(index='Days in Office', columns='President', values='Approving')
pres_pivot.head()

plot_kwargs = dict(figsize=(16,6), color=cm.gray([.3, .7]), style=['-', '--'], title='Approval Rating')
pres_pivot.loc[:250, ['Donald J. Trump', 'Barack Obama']].ffill().plot(**plot_kwargs)

pres_rm = pres_41_45.groupby('President', sort=False)                     .rolling('90D', on='End Date')['Approving']                     .mean()
pres_rm.head()

styles = ['-.', '-', ':', '-', ':']
colors = [.9, .3, .7, .3, .9]
color = cm.Greys(colors)
title='90 Day Approval Rating Rolling Average'
plot_kwargs = dict(figsize=(16,6), style=styles, color = color, title=title)
correct_col_order = pres_41_45.President.unique()
pres_rm.unstack('President')[correct_col_order].plot(**plot_kwargs)

from IPython.display import display_html

years = 2016, 2017, 2018
stock_tables = [pd.read_csv('data/stocks_{}.csv'.format(year), index_col='Symbol') 
                for year in years]

def display_frames(frames, num_spaces=0):
    t_style = '<table style="display: inline;"'
    tables_html = [df.to_html().replace('<table', t_style) for df in frames]

    space = '&nbsp;' * num_spaces
    display_html(space.join(tables_html), raw=True)

display_frames(stock_tables, 30)
stocks_2016, stocks_2017, stocks_2018 = stock_tables

pd.concat(stock_tables, keys=[2016, 2017, 2018])

pd.concat(dict(zip(years,stock_tables)), axis='columns')

stocks_2016.join(stocks_2017, lsuffix='_2016', rsuffix='_2017', how='outer')

stocks_2016

other = [stocks_2017.add_suffix('_2017'), stocks_2018.add_suffix('_2018')]
stocks_2016.add_suffix('_2016').join(other, how='outer')

stock_join = stocks_2016.add_suffix('_2016').join(other, how='outer')
stock_concat = pd.concat(dict(zip(years,stock_tables)), axis='columns')

stock_concat.columns = stock_concat.columns.get_level_values(1) + '_' +                             stock_concat.columns.get_level_values(0).astype(str)

stock_concat

step1 = stocks_2016.merge(stocks_2017, left_index=True, right_index=True, 
                          how='outer', suffixes=('_2016', '_2017'))
stock_merge = step1.merge(stocks_2018.add_suffix('_2018'), 
                          left_index=True, right_index=True, how='outer')

stock_concat.equals(stock_merge)

names = ['prices', 'transactions']
food_tables = [pd.read_csv('data/food_{}.csv'.format(name)) for name in names]
food_prices, food_transactions = food_tables
display_frames(food_tables, 30)

food_transactions.merge(food_prices, on=['item', 'store'])

food_transactions.merge(food_prices.query('Date == 2017'), how='left')

food_prices_join = food_prices.query('Date == 2017').set_index(['item', 'store'])
food_prices_join

food_transactions.join(food_prices_join, on=['item', 'store'])

pd.concat([food_transactions.set_index(['item', 'store']), 
           food_prices.set_index(['item', 'store'])], axis='columns')

import glob

df_list = []
for filename in glob.glob('data/gas prices/*.csv'):
    df_list.append(pd.read_csv(filename, index_col='Week', parse_dates=['Week']))

gas = pd.concat(df_list, axis='columns')
gas.head()

from sqlalchemy import create_engine
engine = create_engine('sqlite:///data/chinook.db')

tracks = pd.read_sql_table('tracks', engine)
tracks.head()

genres = pd.read_sql_table('genres', engine)
genres.head()

genre_track = genres.merge(tracks[['GenreId', 'Milliseconds']], 
                           on='GenreId', how='left') \
                     .drop('GenreId', axis='columns')
genre_track.head()

genre_time = genre_track.groupby('Name')['Milliseconds'].mean()
pd.to_timedelta(genre_time, unit='ms').dt.floor('s').sort_values()

cust = pd.read_sql_table('customers', engine, 
                         columns=['CustomerId', 'FirstName', 'LastName'])
invoice = pd.read_sql_table('invoices', engine, 
                            columns=['InvoiceId','CustomerId'])
ii = pd.read_sql_table('invoice_items', engine, 
                       columns=['InvoiceId', 'UnitPrice', 'Quantity'])

cust_inv = cust.merge(invoice, on='CustomerId')                .merge(ii, on='InvoiceId')
cust_inv.head()

total = cust_inv['Quantity'] * cust_inv['UnitPrice']
cols = ['CustomerId', 'FirstName', 'LastName']
cust_inv.assign(Total = total).groupby(cols)['Total']                                   .sum()                                   .sort_values(ascending=False).head()

pd.read_sql_query('select * from tracks limit 5', engine)

sql_string1 = '''
select 
    Name, 
    time(avg(Milliseconds) / 1000, 'unixepoch') as avg_time
from (
        select 
            g.Name, 
            t.Milliseconds
        from 
            genres as g 
        join
            tracks as t
            on 
                g.genreid == t.genreid
    )
group by 
    Name
order by 
    avg_time
'''
pd.read_sql_query(sql_string1, engine)

sql_string2 = '''
select 
      c.customerid, 
      c.FirstName, 
      c.LastName, 
      sum(ii.quantity *  ii.unitprice) as Total
from
     customers as c
join
     invoices as i
          on c.customerid = i.customerid
join
    invoice_items as ii
          on i.invoiceid = ii.invoiceid
group by
    c.customerid, c.FirstName, c.LastName
order by
    Total desc
'''
pd.read_sql_query(sql_string2, engine)



