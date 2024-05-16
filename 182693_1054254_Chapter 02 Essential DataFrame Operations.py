import pandas as pd
import numpy as np
pd.options.display.max_columns = 40

movie = pd.read_csv('data/movie.csv')
movie_actor_director = movie[['actor_1_name', 'actor_2_name', 'actor_3_name', 'director_name']]
movie_actor_director.head()

movie[['director_name']].head()

movie['actor_1_name', 'actor_2_name', 'actor_3_name', 'director_name']

cols =['actor_1_name', 'actor_2_name', 'actor_3_name', 'director_name']
movie_actor_director = movie[cols]

movie = pd.read_csv('data/movie.csv', index_col='movie_title')
movie.get_dtype_counts()

movie.select_dtypes(include=['int']).head()

movie.select_dtypes(include=['number']).head()

movie.filter(like='facebook').head()

movie.filter(regex='\d').head()

movie.filter(items=['actor_1_name', 'asdf']).head()

movie = pd.read_csv('data/movie.csv')

movie.head()

movie.columns

disc_core = ['movie_title','title_year', 'content_rating','genres']
disc_people = ['director_name','actor_1_name', 'actor_2_name','actor_3_name']
disc_other = ['color','country','language','plot_keywords','movie_imdb_link']
cont_fb = ['director_facebook_likes','actor_1_facebook_likes','actor_2_facebook_likes',
           'actor_3_facebook_likes', 'cast_total_facebook_likes', 'movie_facebook_likes']
cont_finance = ['budget','gross']
cont_num_reviews = ['num_voted_users','num_user_for_reviews', 'num_critic_for_reviews']
cont_other = ['imdb_score','duration', 'aspect_ratio', 'facenumber_in_poster']

new_col_order = disc_core + disc_people + disc_other +                     cont_fb + cont_finance + cont_num_reviews + cont_other
set(movie.columns) == set(new_col_order)

movie2 = movie[new_col_order]
movie2.head()

pd.options.display.max_rows = 8
movie = pd.read_csv('data/movie.csv')
movie.shape

movie.size

movie.ndim

len(movie)

movie.count()

movie.min()

movie.describe()

pd.options.display.max_rows = 10

movie.describe(percentiles=[.01, .3, .99])

pd.options.display.max_rows = 8

movie.isnull().sum()

movie.min(skipna=False)

movie = pd.read_csv('data/movie.csv')
movie.isnull().head()

movie.isnull().sum().head()

movie.isnull().sum().sum()

movie.isnull().any().any()

movie.isnull().get_dtype_counts()

movie[['color', 'movie_title', 'color']].max()

movie.select_dtypes(['object']).fillna('').max()

college = pd.read_csv('data/college.csv')
college + 5

college = pd.read_csv('data/college.csv', index_col='INSTNM')
college_ugds_ = college.filter(like='UGDS_')

college == 'asdf'

college_ugds_.head()

college_ugds_.head() + .00501

(college_ugds_.head() + .00501) // .01

college_ugds_op_round = (college_ugds_ + .00501) // .01 / 100
college_ugds_op_round.head()

college_ugds_round = (college_ugds_ + .00001).round(2)
college_ugds_round.head()

.045 + .005

college_ugds_op_round.equals(college_ugds_round)

college_ugds_op_round_methods = college_ugds_.add(.00501).floordiv(.01).div(100)

np.nan == np.nan

None == None

5 > np.nan

np.nan > 5

5 != np.nan

college = pd.read_csv('data/college.csv', index_col='INSTNM')
college_ugds_ = college.filter(like='UGDS_')

college_ugds_.head() == .0019

college_self_compare = college_ugds_ == college_ugds_
college_self_compare.head()

college_self_compare.all()

(college_ugds_ == np.nan).sum()

college_ugds_.isnull().sum()

from pandas.testing import assert_frame_equal

assert_frame_equal(college_ugds_, college_ugds_)

college_ugds_.eq(.0019).head()

college = pd.read_csv('data/college.csv', index_col='INSTNM')
college_ugds_ = college.filter(like='UGDS_')
college_ugds_.head()

college_ugds_.count()

college_ugds_.count(axis=0)

college_ugds_.count(axis='index')

college_ugds_.count(axis='columns').head()

college_ugds_.sum(axis='columns').head()

college_ugds_.median(axis='index')

college_ugds_cumsum = college_ugds_.cumsum(axis=1)
college_ugds_cumsum.head()

college_ugds_cumsum.sort_values('UGDS_HISP', ascending=False)

pd.read_csv('data/college_diversity.csv', index_col='School')

college = pd.read_csv('data/college.csv', index_col='INSTNM')
college_ugds_ = college.filter(like='UGDS_')
college_ugds_.head()

college_ugds_.isnull().sum(axis=1).sort_values(ascending=False).head()

college_ugds_ = college_ugds_.dropna(how='all')

college_ugds_.isnull().sum()

college_ugds_.ge(.15).head()

diversity_metric = college_ugds_.ge(.15).sum(axis='columns')
diversity_metric.head()

diversity_metric.value_counts()

diversity_metric.sort_values(ascending=False).head()

college_ugds_.loc[['Regency Beauty Institute-Austin', 
                          'Central Texas Beauty College-Temple']]

us_news_top = ['Rutgers University-Newark', 
               'Andrews University', 
               'Stanford University', 
               'University of Houston',
               'University of Nevada-Las Vegas']

diversity_metric.loc[us_news_top]

college_ugds_.max(axis=1).sort_values(ascending=False).head(10)

college_ugds_.loc['Talmudical Seminary Oholei Torah']

(college_ugds_ > .01).all(axis=1).any()

