import pandas as pd
import numpy as np

ted_data = pd.read_csv("ted_main.csv")
ted_data.head(3)

# Let's have a look how many values are missing.
ted_data.isnull().sum()

#Lets have a look at the data and see identify Object/Categorial values and Continuous values
ted_data.dtypes

#Drop the name column
ted_data = ted_data.drop(['name'], axis = 1)
ted_data.columns

from datetime import datetime
def convert(x):
    return pd.to_datetime(x,unit='s')

ted_data['film_date'] = ted_data['film_date'].apply(convert)
ted_data['published_date'] = ted_data['published_date'].apply(convert)
ted_data.head()

#Lets see who talked a lot - top 20
import seaborn as sns
ax = sns.barplot(x="duration", y="main_speaker", data=ted_data.sort_values('duration', ascending=False)[:20])

#Let's see which video got the most views
ax = sns.barplot(x="views", y="main_speaker", data=ted_data.sort_values('views', ascending=False)[:20])

#let's see the distribution of views
sns.distplot(ted_data[ted_data['views'] < 0.4e7]['views'])

#let's see the distribution of duration
sns.distplot(ted_data[ted_data['duration'] < 0.4e7]['duration'])

ax = sns.jointplot(x='views', y='duration', data=ted_data)

#Lets see the ditribution of comments.
sns.distplot(ted_data[ted_data['comments'] < 500]['comments'])

sns.jointplot(x='views', y='comments', data=ted_data)

ted_data[['title', 'main_speaker','views', 'comments', 'duration']].sort_values('views', ascending=False).head(20)

ted_data.head(1)

talk_month = pd.DataFrame(ted_data['film_date'].map(lambda x: x.month).value_counts()).reset_index()
talk_month.columns = ['month', 'talks']
talk_month.head()

sns.barplot(x='month', y='talks', data=talk_month)

