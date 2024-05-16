import tweepy
import codecs
from time import sleep

## fill in your Twitter credentials 
access_token = ''
access_token_secret = ''
consumer_key = ''
consumer_secret = ''

## Set up an instance of the REST API
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

#stuff = api.user_timeline(screen_name = 'Inspire_Us', count = 100)

#for status in stuff:
#    print(status.text)

for tweet in tweepy.Cursor(api.search, q='#motivation').items():
    try:
        print('\nTweet by: @' + tweet.user.screen_name)

        tweet.retweet()
        print('Retweeted the tweet')

        sleep(3600)

    except tweepy.TweepError as e:
        print(e.reason)

    except StopIteration:
        break

api.update_status(status="This is a sample tweet using Tweepy with python")

for tweet in tweepy.Cursor(api.search, q='#motivation').items():
    try:
        print('\nTweet by: @' + tweet.user.screen_name)

        # Favorite the tweet
        tweet.favorite()
        print('Favorited the tweet')

        # Follow the user who tweeted
        tweet.user.follow()
        print('Followed the user')

        #sleep(5)
        sleep(3600)

    except tweepy.TweepError as e:
        print(e.reason)

    except StopIteration:
        break       
        

