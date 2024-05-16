get_ipython().magic('matplotlib inline')

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

tweets = pd.read_json("https://s3.amazonaws.com/far-right/twitter/mb_protests.json")
tweets

from pprint import pprint 
print('Top Tweeters')
user_tweet_counts = tweets['user_name'].value_counts()
pprint(user_tweet_counts[:20])

hi_tweeters = user_tweet_counts[user_tweet_counts > 20]
plt.title('High volume tweeters tweet counts')
plt.hist(hi_tweeters, bins = 20)
plt.show()

lo_tweeters = user_tweet_counts[user_tweet_counts <= 20]
plt.title('Lo volume tweeters tweet counts')
plt.hist(lo_tweeters, bins = 20)
plt.show()

def total_tweet_count(tweets):
    return len(tweets)

def unique_tweets(tweets):
    return tweets['text'].unique()

def unique_tweet_count(tweets):
    return len(unique_tweets(tweets))

def unique_users(tweets):
    return tweets['user_name'].unique()

def unique_users_count(tweets):
    return len(unique_users(tweets))

print(str(total_tweet_count(tweets)) + " total tweets")

print(str(unique_tweet_count(tweets)) + " unique tweets")

print(str(unique_users_count(tweets)) + " unique user_names")

def hashtags_table(tweets):
    hashtags= {}
    for row in tweets['hashtags'].unique():
        row = eval(row)
        for tag in row:
            tag = tag.lower()
            if tag not in hashtags:
                hashtags[tag] = 1
            else:
                hashtags[tag] = hashtags[tag] + 1
    hashtagspd = pd.DataFrame(list(hashtags.items()), columns=['hashtag', 'count']).sort_values('count', ascending=False)
    return hashtagspd
hashtags = hashtags_table(tweets)
print("Top 20 hashtags")
pprint(hashtags[:20])



