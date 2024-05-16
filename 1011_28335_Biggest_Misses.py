import psycopg2
import pandas.io.sql as psql
import pandas as pd
from matplotlib import pyplot as plt
from __future__ import division #now division always returns a floating point number
import numpy as np
import seaborn as sns
get_ipython().magic('matplotlib inline')

db = psycopg2.connect("dbname='fantasyfootball' host='localhost'")

def get_combined_df():
    actual_points = psql.read_sql("""
    select name, team, position, sum(total_points) as points_actual 
    from scoring_leaders_weekly
    group by name, team, position;""", con=db)
    predicted_points = psql.read_sql("""
        select name, team, position, sum(total_points) as points_predicted
        from next_week_projections
        group by name, team, position;""", con=db)
    combined_df = actual_points.merge(predicted_points, 
                                      on=['name', 'team', 'position'], how='left')
    combined_df = combined_df.dropna()
    combined_df = combined_df[combined_df['points_predicted'] > 0]
    combined_df['points_diff'] = combined_df.points_actual - combined_df.points_predicted
    combined_df['points_diff_pct'] = (combined_df.points_actual - combined_df.points_predicted) / combined_df.points_predicted
    return combined_df

def get_top_bottom(df):
    group = df.groupby('position')
    top_list = []
    bottom_list = []
    for name, data in group:
        top = data.sort('points_diff', ascending=False)
        top_list.append(top.head())
        tail = top.tail()
        tail = tail.sort('points_diff')
        bottom_list.append(tail)
    top_df = pd.concat(top_list)
    bottom_df = pd.concat(bottom_list)
    return top_df, bottom_df

def run_analysis():
    combined_df = get_combined_df()
    top, bottom = get_top_bottom(combined_df)
    return combined_df, top, bottom

combined_df, top_1, bottom_1 = run_analysis()

top_1

bottom_1

ax = sns.boxplot(combined_df.points_diff, groupby=combined_df.position)
plt.title("Distribution of Point Differences by Position")
sns.despine()

ax = sns.boxplot(combined_df.points_actual, groupby=combined_df.position)
plt.title("Distribution of Actual Points by Position")
sns.despine()

combined_df[combined_df['position'] == "QB"].sort('points_actual', ascending=False).head(n=12).describe()

combined_df[combined_df['position'] == "QB"].sort('points_actual', ascending=False).head(n=36).describe()

combined_df.describe()

