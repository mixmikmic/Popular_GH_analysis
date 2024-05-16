import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')

pd.options.display.max_columns = 50

movie = pd.read_csv('data/movie.csv', index_col='movie_title')
movie.head()

movie_2_hours = movie['duration'] > 120
movie_2_hours.head(10)

movie_2_hours.sum()

movie_2_hours.mean()

movie_2_hours.describe()

movie['duration'].dropna().gt(120).mean()

movie_2_hours.value_counts(normalize=True)

actors = movie[['actor_1_facebook_likes', 'actor_2_facebook_likes']].dropna()
(actors['actor_1_facebook_likes'] > actors['actor_2_facebook_likes']).mean()

movie = pd.read_csv('data/movie.csv', index_col='movie_title')
movie.head()

criteria1 = movie.imdb_score > 8
criteria2 = movie.content_rating == 'PG-13'
criteria3 = (movie.title_year < 2000) | (movie.title_year >= 2010)

criteria2.head()

criteria_final = criteria1 & criteria2 & criteria3
criteria_final.head()

movie.title_year < 2000 | movie.title_year > 2009

movie = pd.read_csv('data/movie.csv', index_col='movie_title')

crit_a1 = movie.imdb_score > 8
crit_a2 = movie.content_rating == 'PG-13'
crit_a3 = (movie.title_year < 2000) | (movie.title_year > 2009)
final_crit_a = crit_a1 & crit_a2 & crit_a3

crit_b1 = movie.imdb_score < 5
crit_b2 = movie.content_rating == 'R'
crit_b3 = (movie.title_year >= 2000) & (movie.title_year <= 2010)
final_crit_b = crit_b1 & crit_b2 & crit_b3

final_crit_all = final_crit_a | final_crit_b
final_crit_all.head()

movie[final_crit_all].head()

cols = ['imdb_score', 'content_rating', 'title_year']
movie_filtered = movie.loc[final_crit_all, cols]
movie_filtered.head(10)

final_crit_a2 = (movie.imdb_score > 8) &                 (movie.content_rating == 'PG-13') &                 ((movie.title_year < 2000) | (movie.title_year > 2009))
final_crit_a2.equals(final_crit_a)

college = pd.read_csv('data/college.csv')
college[college['STABBR'] == 'TX'].head()

college2 = college.set_index('STABBR')
college2.loc['TX'].head()

get_ipython().run_line_magic('timeit', "college[college['STABBR'] == 'TX']")

get_ipython().run_line_magic('timeit', "college2.loc['TX']")

get_ipython().run_line_magic('timeit', "college2 = college.set_index('STABBR')")

states =['TX', 'CA', 'NY']
college[college['STABBR'].isin(states)]
college2.loc[states].head()

college = pd.read_csv('data/college.csv')
college2 = college.set_index('STABBR')

college2.index.is_monotonic

college3 = college2.sort_index()
college3.index.is_monotonic

get_ipython().run_line_magic('timeit', "college[college['STABBR'] == 'TX']")

get_ipython().run_line_magic('timeit', "college2.loc['TX']")

get_ipython().run_line_magic('timeit', "college3.loc['TX']")

college_unique = college.set_index('INSTNM')
college_unique.index.is_unique

college[college['INSTNM'] == 'Stanford University']

college_unique.loc['Stanford University']

get_ipython().run_line_magic('timeit', "college[college['INSTNM'] == 'Stanford University']")

get_ipython().run_line_magic('timeit', "college_unique.loc['Stanford University']")

college.index = college['CITY'] + ', ' + college['STABBR']
college = college.sort_index()
college.head()

college.loc['Miami, FL'].head()

get_ipython().run_cell_magic('timeit', '', "crit1 = college['CITY'] == 'Miami' \ncrit2 = college['STABBR'] == 'FL'\ncollege[crit1 & crit2]")

get_ipython().run_line_magic('timeit', "college.loc['Miami, FL']")

college[(college['CITY'] == 'Miami') & (college['STABBR'] == 'FL')].equals(college.loc['Miami, FL'])

slb = pd.read_csv('data/slb_stock.csv', index_col='Date', parse_dates=['Date'])
slb.head()

slb_close = slb['Close']
slb_summary = slb_close.describe(percentiles=[.1, .9])
slb_summary

upper_10 = slb_summary.loc['90%']
lower_10 = slb_summary.loc['10%']
criteria = (slb_close < lower_10) | (slb_close > upper_10)
slb_top_bottom_10 = slb_close[criteria]

slb_close.plot(color='black', figsize=(12,6))
slb_top_bottom_10.plot(marker='o', style=' ', ms=4, color='lightgray')

xmin = criteria.index[0]
xmax = criteria.index[-1]
plt.hlines(y=[lower_10, upper_10], xmin=xmin, xmax=xmax,color='black')

slb_close.plot(color='black', figsize=(12,6))
plt.hlines(y=[lower_10, upper_10], 
           xmin=xmin, xmax=xmax,color='lightgray')
plt.fill_between(x=criteria.index, y1=lower_10,
                 y2=slb_close.values, color='black')
plt.fill_between(x=criteria.index,y1=lower_10,
                 y2=slb_close.values, where=slb_close < lower_10,
                 color='lightgray')
plt.fill_between(x=criteria.index, y1=upper_10, 
                 y2=slb_close.values, where=slb_close > upper_10,
                 color='lightgray')

employee = pd.read_csv('data/employee.csv')

employee.DEPARTMENT.value_counts().head()

employee.GENDER.value_counts()

employee.BASE_SALARY.describe().astype(int)

depts = ['Houston Police Department-HPD', 
             'Houston Fire Department (HFD)']
criteria_dept = employee.DEPARTMENT.isin(depts)
criteria_gender = employee.GENDER == 'Female'
criteria_sal = (employee.BASE_SALARY >= 80000) &                (employee.BASE_SALARY <= 120000)

criteria_final = criteria_dept & criteria_gender & criteria_sal

select_columns = ['UNIQUE_ID', 'DEPARTMENT', 'GENDER', 'BASE_SALARY']
employee.loc[criteria_final, select_columns].head()

criteria_sal = employee.BASE_SALARY.between(80000, 120000)

top_5_depts = employee.DEPARTMENT.value_counts().index[:5]
criteria = ~employee.DEPARTMENT.isin(top_5_depts)
employee[criteria].head()

amzn = pd.read_csv('data/amzn_stock.csv', index_col='Date', parse_dates=['Date'])
amzn.head()

amzn_daily_return = amzn.Close.pct_change()
amzn_daily_return.head()

amzn_daily_return = amzn_daily_return.dropna()
amzn_daily_return.hist(bins=20)

mean = amzn_daily_return.mean()  
std = amzn_daily_return.std()

abs_z_score = amzn_daily_return.sub(mean).abs().div(std)

pcts = [abs_z_score.lt(i).mean() for i in range(1,4)]
print('{:.3f} fall within 1 standard deviation. '
      '{:.3f} within 2 and {:.3f} within 3'.format(*pcts))

def test_return_normality(stock_data):
    close = stock_data['Close']
    daily_return = close.pct_change().dropna()
    daily_return.hist(bins=20)
    mean = daily_return.mean() 
    std = daily_return.std()
    
    abs_z_score = abs(daily_return - mean) / std
    pcts = [abs_z_score.lt(i).mean() for i in range(1,4)]

    print('{:.3f} fall within 1 standard deviation. '
          '{:.3f} within 2 and {:.3f} within 3'.format(*pcts))

slb = pd.read_csv('data/slb_stock.csv', 
                  index_col='Date', parse_dates=['Date'])
test_return_normality(slb)

employee = pd.read_csv('data/employee.csv')
depts = ['Houston Police Department-HPD', 'Houston Fire Department (HFD)']
select_columns = ['UNIQUE_ID', 'DEPARTMENT', 'GENDER', 'BASE_SALARY']

qs = "DEPARTMENT in @depts "          "and GENDER == 'Female' "          "and 80000 <= BASE_SALARY <= 120000"
        
emp_filtered = employee.query(qs)
emp_filtered[select_columns].head()

top10_depts = employee.DEPARTMENT.value_counts().index[:10].tolist()
qs = "DEPARTMENT not in @top10_depts and GENDER == 'Female'"
employee_filtered2 = employee.query(qs)
employee_filtered2[['DEPARTMENT', 'GENDER']].head()

movie = pd.read_csv('data/movie.csv', index_col='movie_title')
fb_likes = movie['actor_1_facebook_likes'].dropna()
fb_likes.head()

fb_likes.describe(percentiles=[.1, .25, .5, .75, .9]).astype(int)

fb_likes.describe(percentiles=[.1,.25,.5,.75,.9])

fb_likes.hist()

criteria_high = fb_likes < 20000
criteria_high.mean().round(2)

fb_likes.where(criteria_high).head()

fb_likes.where(criteria_high, other=20000).head()

criteria_low = fb_likes > 300
fb_likes_cap = fb_likes.where(criteria_high, other=20000)                       .where(criteria_low, 300)
fb_likes_cap.head()

len(fb_likes), len(fb_likes_cap)

fb_likes_cap.hist()

fb_likes_cap2 = fb_likes.clip(lower=300, upper=20000)
fb_likes_cap2.equals(fb_likes_cap)

movie = pd.read_csv('data/movie.csv', index_col='movie_title')
c1 = movie['title_year'] >= 2010
c2 = movie['title_year'].isnull()
criteria = c1 | c2

movie.mask(criteria).head()

movie_mask = movie.mask(criteria).dropna(how='all')
movie_mask.head()

movie_boolean = movie[movie['title_year'] < 2010]
movie_boolean.head()

movie_mask.equals(movie_boolean)

movie_mask.shape == movie_boolean.shape

movie_mask.dtypes == movie_boolean.dtypes

from pandas.testing import assert_frame_equal
assert_frame_equal(movie_boolean, movie_mask, check_dtype=False)

get_ipython().run_line_magic('timeit', "movie.mask(criteria).dropna(how='all')")

get_ipython().run_line_magic('timeit', "movie[movie['title_year'] < 2010]")

movie = pd.read_csv('data/movie.csv', index_col='movie_title')
c1 = movie['content_rating'] == 'G'
c2 = movie['imdb_score'] < 4
criteria = c1 & c2

movie_loc = movie.loc[criteria]
movie_loc.head()

movie_loc.equals(movie[criteria])

movie_iloc = movie.iloc[criteria]

movie_iloc = movie.iloc[criteria.values]

movie_iloc.equals(movie_loc)

movie.loc[criteria.values]

criteria_col = movie.dtypes == np.int64
criteria_col.head()

movie.loc[:, criteria_col].head()

movie.iloc[:, criteria_col.values].head()

cols = ['content_rating', 'imdb_score', 'title_year', 'gross']
movie.loc[criteria, cols].sort_values('imdb_score')

col_index = [movie.columns.get_loc(col) for col in cols]
col_index

movie.iloc[criteria.values, col_index].sort_values('imdb_score')

a = criteria.values
a[:5]

len(a), len(criteria)

movie.loc[[True, False, True], [True, False, False, True]]



