import pandas as pd
import numpy as np
from IPython.display import display
pd.options.display.max_columns = 50

college = pd.read_csv('data/college.csv')

college.head()

college.shape

with pd.option_context('display.max_rows', 8):
    display(college.describe(include=[np.number]).T)

college.describe(include=[np.object, pd.Categorical]).T

college.info()

college.describe(include=[np.number]).T

college.describe(include=[np.object, pd.Categorical]).T

with pd.option_context('display.max_rows', 5):
    display(college.describe(include=[np.number], 
                 percentiles=[.01, .05, .10, .25, .5, .75, .9, .95, .99]).T)

college_dd = pd.read_csv('data/college_data_dictionary.csv')

with pd.option_context('display.max_rows', 8):
    display(college_dd)

college = pd.read_csv('data/college.csv')
different_cols = ['RELAFFIL', 'SATMTMID', 'CURROPER', 'INSTNM', 'STABBR']
col2 = college.loc[:, different_cols]
col2.head()

col2.dtypes

original_mem = col2.memory_usage(deep=True)
original_mem

col2['RELAFFIL'] = col2['RELAFFIL'].astype(np.int8)

col2.dtypes

col2.select_dtypes(include=['object']).nunique()

col2['STABBR'] = col2['STABBR'].astype('category')
col2.dtypes

new_mem = col2.memory_usage(deep=True)
new_mem

new_mem / original_mem

college = pd.read_csv('data/college.csv')

college[['CURROPER', 'INSTNM']].memory_usage(deep=True)

college.loc[0, 'CURROPER'] = 10000000
college.loc[0, 'INSTNM'] = college.loc[0, 'INSTNM'] + 'a'
# college.loc[1, 'INSTNM'] = college.loc[1, 'INSTNM'] + 'a'
college[['CURROPER', 'INSTNM']].memory_usage(deep=True)

college['MENONLY'].dtype

college['MENONLY'].astype('int8') # ValueError: Cannot convert non-finite values (NA or inf) to integer

college.describe(include=['int64', 'float64']).T

college.describe(include=[np.int64, np.float64]).T

college['RELAFFIL'] = college['RELAFFIL'].astype(np.int8)

college.describe(include=['int', 'float']).T  # defaults to 64 bit int/floats

college.describe(include=['number']).T  # also works as the default int/float are 64 bits

college['MENONLY'] = college['MENONLY'].astype('float16')
college['RELAFFIL'] = college['RELAFFIL'].astype('int8')

college.index = pd.Int64Index(college.index)
college.index.memory_usage()

movie = pd.read_csv('data/movie.csv')
movie2 = movie[['movie_title', 'imdb_score', 'budget']]
movie2.head()

movie2.nlargest(100, 'imdb_score').head()

movie2.nlargest(100, 'imdb_score').nsmallest(5, 'budget')

movie = pd.read_csv('data/movie.csv')
movie2 = movie[['movie_title', 'title_year', 'imdb_score']]

movie2.sort_values('title_year', ascending=False).head()

movie3 = movie2.sort_values(['title_year','imdb_score'], ascending=False)
movie3.head()

movie_top_year = movie3.drop_duplicates(subset='title_year')
movie_top_year.head()

movie4 = movie[['movie_title', 'title_year', 'content_rating', 'budget']]
movie4_sorted = movie4.sort_values(['title_year', 'content_rating', 'budget'], 
                                   ascending=[False, False, True])
movie4_sorted.drop_duplicates(subset=['title_year', 'content_rating']).head(10)

movie = pd.read_csv('data/movie.csv')
movie2 = movie[['movie_title', 'imdb_score', 'budget']]
movie_smallest_largest = movie2.nlargest(100, 'imdb_score').nsmallest(5, 'budget')
movie_smallest_largest

movie2.sort_values('imdb_score', ascending=False).head(100).head()

movie2.sort_values('imdb_score', ascending=False).head(100).sort_values('budget').head()

movie2.nlargest(100, 'imdb_score').tail()

movie2.sort_values('imdb_score', ascending=False).head(100).tail()

import pandas_datareader as pdr

tsla = pdr.DataReader('tsla', data_source='yahoo',start='2017-1-1')
tsla.head(8)

tsla_close = tsla['Close']

tsla_cummax = tsla_close.cummax()
tsla_cummax.head(8)

tsla_trailing_stop = tsla_cummax * .9
tsla_trailing_stop.head(8)

def set_trailing_loss(symbol, purchase_date, perc):
    close = pdr.DataReader(symbol, 'yahoo', start=purchase_date)['Close']
    return close.cummax() * perc

msft_trailing_stop = set_trailing_loss('msft', '2017-6-1', .85)
msft_trailing_stop.head()



