import pandas as pd
import numpy as np
import datetime

get_ipython().run_line_magic('matplotlib', 'inline')

date = datetime.date(year=2013, month=6, day=7)
time = datetime.time(hour=12, minute=30, second=19, microsecond=463198)
dt = datetime.datetime(year=2013, month=6, day=7, 
                       hour=12, minute=30, second=19, microsecond=463198)

print("date is ", date)
print("time is", time)
print("datetime is", dt)

td = datetime.timedelta(weeks=2, days=5, hours=10, minutes=20, 
                        seconds=6.73, milliseconds=99, microseconds=8)
print(td)

print('new date is', date + td)
print('new datetime is', dt + td)

time + td

pd.Timestamp(year=2012, month=12, day=21, hour=5, minute=10, second=8, microsecond=99)

pd.Timestamp('2016/1/10')

pd.Timestamp('2014-5/10')

pd.Timestamp('Jan 3, 2019 20:45.56')

pd.Timestamp('2016-01-05T05:34:43.123456789')

pd.Timestamp(500)

pd.Timestamp(5000, unit='D')

pd.to_datetime('2015-5-13')

pd.to_datetime('2015-13-5', dayfirst=True)

pd.Timestamp('Saturday September 30th, 2017')

pd.to_datetime('Start Date: Sep 30, 2017 Start Time: 1:30 pm', format='Start Date: %b %d, %Y Start Time: %I:%M %p')

pd.to_datetime(100, unit='D', origin='2013-1-1')

s = pd.Series([10, 100, 1000, 10000])
pd.to_datetime(s, unit='D')

s = pd.Series(['12-5-2015', '14-1-2013', '20/12/2017', '40/23/2017'])
pd.to_datetime(s, dayfirst=True, errors='coerce')

pd.to_datetime(['Aug 3 1999 3:45:56', '10/31/2017'])

pd.Timedelta('12 days 5 hours 3 minutes 123456789 nanoseconds')

pd.Timedelta(days=5, minutes=7.34)

pd.Timedelta(100, unit='W')

pd.to_timedelta('5 dayz', errors='ignore')

pd.to_timedelta('67:15:45.454')

s = pd.Series([10, 100])
pd.to_timedelta(s, unit='s')

time_strings = ['2 days 24 minutes 89.67 seconds', '00:45:23.6']
pd.to_timedelta(time_strings)

pd.Timedelta('12 days 5 hours 3 minutes') * 2

pd.Timestamp('1/1/2017') + pd.Timedelta('12 days 5 hours 3 minutes') * 2

td1 = pd.to_timedelta([10, 100], unit='s')
td2 = pd.to_timedelta(['3 hours', '4 hours'])
td1 + td2

pd.Timedelta('12 days') / pd.Timedelta('3 days')

ts = pd.Timestamp('2016-10-1 4:23:23.9')

ts.ceil('h')

ts.year, ts.month, ts.day, ts.hour, ts.minute, ts.second

ts.dayofweek, ts.dayofyear, ts.daysinmonth

ts.to_pydatetime()

td = pd.Timedelta(125.8723, unit='h')
td

td.round('min')

td.components

td.total_seconds()

date_string_list = ['Sep 30 1984'] * 10000

get_ipython().run_line_magic('timeit', "pd.to_datetime(date_string_list, format='%b %d %Y')")

get_ipython().run_line_magic('timeit', 'pd.to_datetime(date_string_list)')

crime = pd.read_hdf('data/crime.h5', 'crime')
crime.dtypes

crime = crime.set_index('REPORTED_DATE')
crime.head()

pd.options.display.max_rows = 4

crime.loc['2016-05-12 16:45:00']

crime.loc['2016-05-12']

crime.loc['2016-05'].shape

crime.loc['2016'].shape

crime.loc['2016-05-12 03'].shape

crime.loc['Dec 2015'].sort_index()

crime.loc['2016 Sep, 15'].shape

crime.loc['21st October 2014 05'].shape

crime.loc['2015-3-4':'2016-1-1'].sort_index()

crime.loc['2015-3-4 22':'2016-1-1 23:45:00'].sort_index()

mem_cat = crime.memory_usage().sum()
mem_obj = crime.astype({'OFFENSE_TYPE_ID':'object', 
                        'OFFENSE_CATEGORY_ID':'object', 
                        'NEIGHBORHOOD_ID':'object'}).memory_usage(deep=True)\
                                                    .sum()
mb = 2 ** 20
round(mem_cat / mb, 1), round(mem_obj / mb, 1)

crime.index[:2]

get_ipython().run_line_magic('timeit', "crime.loc['2015-3-4':'2016-1-1']")

crime_sort = crime.sort_index()

get_ipython().run_line_magic('timeit', "crime_sort.loc['2015-3-4':'2016-1-1']")

pd.options.display.max_rows = 60

crime = pd.read_hdf('data/crime.h5', 'crime').set_index('REPORTED_DATE')
print(type(crime.index))

crime.between_time('2:00', '5:00', include_end=False).head()

crime.at_time('5:47').head()

crime_sort = crime.sort_index()

pd.options.display.max_rows = 6

crime_sort.first(pd.offsets.MonthBegin(6))

crime_sort.first(pd.offsets.MonthEnd(6))

crime_sort.first(pd.offsets.MonthBegin(6, normalize=True))

crime_sort.loc[:'2012-06']

crime_sort.first('5D')

crime_sort.first('5B')

crime_sort.first('7W')

crime_sort.first('3QS')

import datetime
crime.between_time(datetime.time(2,0), datetime.time(5,0), include_end=False)

first_date = crime_sort.index[0]
first_date

first_date + pd.offsets.MonthBegin(6)

first_date + pd.offsets.MonthEnd(6)

dt = pd.Timestamp('2012-1-16 13:40')
dt + pd.DateOffset(months=1)

do = pd.DateOffset(years=2, months=5, days=3, hours=8, seconds=10)
pd.Timestamp('2012-1-22 03:22') + do

pd.options.display.max_rows=60

crime_sort = pd.read_hdf('data/crime.h5', 'crime')                .set_index('REPORTED_DATE')                .sort_index()

crime_sort.resample('W')

weekly_crimes = crime_sort.resample('W').size()
weekly_crimes.head()

len(crime_sort.loc[:'2012-1-8'])

len(crime_sort.loc['2012-1-9':'2012-1-15'])

crime_sort.resample('W-THU').size().head()

weekly_crimes_gby = crime_sort.groupby(pd.Grouper(freq='W')).size()
weekly_crimes_gby.head()

weekly_crimes.equals(weekly_crimes_gby)

r = crime_sort.resample('W')
resample_methods = [attr for attr in dir(r) if attr[0].islower()]
print(resample_methods)

crime = pd.read_hdf('data/crime.h5', 'crime')
weekly_crimes2 = crime.resample('W', on='REPORTED_DATE').size()
weekly_crimes2.equals(weekly_crimes)

weekly_crimes_gby2 = crime.groupby(pd.Grouper(key='REPORTED_DATE', freq='W')).size()
weekly_crimes_gby2.equals(weekly_crimes_gby)

weekly_crimes.plot(figsize=(16,4), title='All Denver Crimes')

crime_sort = pd.read_hdf('data/crime.h5', 'crime')                .set_index('REPORTED_DATE')                .sort_index()

crime_quarterly = crime_sort.resample('Q')['IS_CRIME', 'IS_TRAFFIC'].sum()
crime_quarterly.head()

crime_sort.resample('QS')['IS_CRIME', 'IS_TRAFFIC'].sum().head()

crime_sort.loc['2012-4-1':'2012-6-30', ['IS_CRIME', 'IS_TRAFFIC']].sum()

crime_quarterly2 = crime_sort.groupby(pd.Grouper(freq='Q'))['IS_CRIME', 'IS_TRAFFIC'].sum()
crime_quarterly2.equals(crime_quarterly)

plot_kwargs = dict(figsize=(16,4), 
                   color=['black', 'lightgrey'], 
                   title='Denver Crimes and Traffic Accidents')
crime_quarterly.plot(**plot_kwargs)

crime_sort.resample('Q').sum().head()

crime_sort.resample('QS-MAR')['IS_CRIME', 'IS_TRAFFIC'].sum().head()

crime_begin = crime_quarterly.iloc[0]
crime_begin

crime_quarterly.div(crime_begin)                .sub(1)                .round(2)                .plot(**plot_kwargs)

crime = pd.read_hdf('data/crime.h5', 'crime')
crime.head()

wd_counts = crime['REPORTED_DATE'].dt.weekday_name.value_counts()
wd_counts

days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 
        'Friday', 'Saturday', 'Sunday']
title = 'Denver Crimes and Traffic Accidents per Weekday'
wd_counts.reindex(days).plot(kind='barh', title=title)

title = 'Denver Crimes and Traffic Accidents per Year' 
crime['REPORTED_DATE'].dt.year.value_counts()                               .sort_index()                               .plot(kind='barh', title=title)

weekday = crime['REPORTED_DATE'].dt.weekday_name
year = crime['REPORTED_DATE'].dt.year

crime_wd_y = crime.groupby([year, weekday]).size()
crime_wd_y.head(10)

crime_table = crime_wd_y.rename_axis(['Year', 'Weekday']).unstack('Weekday')
crime_table

criteria = crime['REPORTED_DATE'].dt.year == 2017
crime.loc[criteria, 'REPORTED_DATE'].dt.dayofyear.max()

round(272 / 365, 3)

crime_pct = crime['REPORTED_DATE'].dt.dayofyear.le(272)                                   .groupby(year)                                   .mean()                                   .round(3)
crime_pct

crime_pct.loc[2012:2016].median()

crime_table.loc[2017] = crime_table.loc[2017].div(.748).astype('int')
crime_table = crime_table.reindex(columns=days)
crime_table

import seaborn as sns
sns.heatmap(crime_table, cmap='Greys')

denver_pop = pd.read_csv('data/denver_pop.csv', index_col='Year')
denver_pop

den_100k = denver_pop.div(100000).squeeze()
crime_table2 = crime_table.div(den_100k, axis='index').astype('int')
crime_table2

sns.heatmap(crime_table2, cmap='Greys')

wd_counts.loc[days]

crime_table / den_100k

ADJ_2017 = .748

def count_crime(df, offense_cat): 
    df = df[df['OFFENSE_CATEGORY_ID'] == offense_cat]
    weekday = df['REPORTED_DATE'].dt.weekday_name
    year = df['REPORTED_DATE'].dt.year
    
    ct = df.groupby([year, weekday]).size().unstack()
    ct.loc[2017] = ct.loc[2017].div(ADJ_2017).astype('int')
    
    pop = pd.read_csv('data/denver_pop.csv', index_col='Year')
    pop = pop.squeeze().div(100000)
    
    ct = ct.div(pop, axis=0).astype('int')
    ct = ct.reindex(columns=days)
    sns.heatmap(ct, cmap='Greys')
    return ct

count_crime(crime, 'auto-theft')

crime_sort = pd.read_hdf('data/crime.h5', 'crime')                .set_index('REPORTED_DATE')                .sort_index()

common_attrs = set(dir(crime_sort.index)) & set(dir(pd.Timestamp))
print([attr for attr in common_attrs if attr[0] != '_'])

crime_sort.index.weekday_name.value_counts()

crime_sort.groupby(lambda x: x.weekday_name)['IS_CRIME', 'IS_TRAFFIC'].sum()

funcs = [lambda x: x.round('2h').hour, lambda x: x.year]
cr_group = crime_sort.groupby(funcs)['IS_CRIME', 'IS_TRAFFIC'].sum()
cr_final = cr_group.unstack()
cr_final.style.highlight_max(color='lightgrey')

cr_final.xs('IS_TRAFFIC', axis='columns', level=0).head()

cr_final.xs(2016, axis='columns', level=1).head()

employee = pd.read_csv('data/employee.csv', 
                       parse_dates=['JOB_DATE', 'HIRE_DATE'], 
                       index_col='HIRE_DATE')
employee.head()

employee.groupby('GENDER')['BASE_SALARY'].mean().round(-2)

employee.resample('10AS')['BASE_SALARY'].mean().round(-2)

sal_avg = employee.groupby('GENDER').resample('10AS')['BASE_SALARY'].mean().round(-2)
sal_avg

sal_avg.unstack('GENDER')

employee[employee['GENDER'] == 'Male'].index.min()

employee[employee['GENDER'] == 'Female'].index.min()

sal_avg2 = employee.groupby(['GENDER', pd.Grouper(freq='10AS')])['BASE_SALARY'].mean().round(-2)
sal_avg2

sal_final = sal_avg2.unstack('GENDER')
sal_final

'resample' in dir(employee.groupby('GENDER'))

'groupby' in dir(employee.resample('10AS'))

years = sal_final.index.year
years_right = years + 9
sal_final.index = years.astype(str) + '-' + years_right.astype(str)
sal_final

cuts = pd.cut(employee.index.year, bins=5, precision=0)
cuts.categories.values

employee.groupby([cuts, 'GENDER'])['BASE_SALARY'].mean().unstack('GENDER').round(-2)

crime_sort = pd.read_hdf('data/crime.h5', 'crime')                .set_index('REPORTED_DATE')                .sort_index()

crime_sort.index.max()

crime_sort = crime_sort[:'2017-8']
crime_sort.index.max()

all_data = crime_sort.groupby([pd.Grouper(freq='M'), 'OFFENSE_CATEGORY_ID']).size()
all_data.head()

all_data = all_data.sort_values().reset_index(name='Total')
all_data.head()

goal = all_data[all_data['REPORTED_DATE'] == '2017-8-31'].reset_index(drop=True)
goal['Total_Goal'] = goal['Total'].mul(.8).astype(int)
goal.head()

pd.merge_asof(goal, all_data, left_on='Total_Goal', right_on='Total', 
              by='OFFENSE_CATEGORY_ID', suffixes=('_Current', '_Last'))

pd.Period(year=2012, month=5, day=17, hour=14, minute=20, freq='T')

crime_sort.index.to_period('M')

ad_period = crime_sort.groupby([lambda x: x.to_period('M'), 
                                'OFFENSE_CATEGORY_ID']).size()
ad_period = ad_period.sort_values()                      .reset_index(name='Total')                      .rename(columns={'level_0':'REPORTED_DATE'})
ad_period.head()

cols = ['OFFENSE_CATEGORY_ID', 'Total']
all_data[cols].equals(ad_period[cols])

aug_2018 = pd.Period('2017-8', freq='M')
goal_period = ad_period[ad_period['REPORTED_DATE'] == aug_2018].reset_index(drop=True)
goal_period['Total_Goal'] = goal_period['Total'].mul(.8).astype(int)

pd.merge_asof(goal_period, ad_period, left_on='Total_Goal', right_on='Total', 
                  by='OFFENSE_CATEGORY_ID', suffixes=('_Current', '_Last')).head()



