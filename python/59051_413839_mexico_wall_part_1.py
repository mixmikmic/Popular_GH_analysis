get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta
import datetime

df = pd.read_json("https://s3.amazonaws.com/far-right/twitter/mb_protests.json")

df.columns

print("Total number of tweets = {}".format(len(df)))

# Lowercase the hashtags and tweet body
df['hashtags'] = df['hashtags'].str.lower()
df['text'] = df['text'].str.lower()

print("Total number of tweets containing hashtag 'wall' = {}".format(len(df[df['hashtags'].str.contains('wall')])))

print("Total number of tweets whose body contains 'wall' = {}".format(len(df[df['text'].str.contains('wall')])))

wall_tweets = df[(df['hashtags'].str.contains('wall')) | (df['text'].str.contains('wall'))].copy()

print("Total number of tweets about the 'wall' = {}".format(len(wall_tweets)))

def months_between(end, start):
    return (end.year - start.year)*12 + end.month - start.month

wall_tweets['created'] = pd.to_datetime(wall_tweets['created'])
wall_tweets['user_created'] = pd.to_datetime(wall_tweets['user_created'])

wall_tweets['user_tenure'] = wall_tweets[['created',                             'user_created']].apply(lambda row: months_between(row[0], row[1]), axis=1)

tenure_grouping = wall_tweets.groupby('user_tenure').size() / len(wall_tweets) * 100

fig, ax = plt.subplots()

ax.plot(tenure_grouping.index, tenure_grouping.values)

ax.set_ylabel("% of tweets")
ax.set_xlabel("Acct tenure in months")

plt.show()

tweets_per_user = wall_tweets.groupby('user_name').size().sort_values(ascending=False)

fig, ax = plt.subplots()

ax.plot(tweets_per_user.values)

plt.show()

wall_tweets.groupby(['user_name', 'user_description']).size().sort_values(ascending=False).head(20).to_frame()

plt.boxplot(wall_tweets['friends_count'].values, vert=False)
plt.show()

wall_tweets['friends_count'].describe()

wall_tweets.groupby('user_location').size().sort_values(ascending=False)



