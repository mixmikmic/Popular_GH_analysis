

import twitter
import urlparse
import pandas as pd

# parallel print
from pprint import pprint as pp

# Load the twitter API keys
twitter_tokens = pd.read_csv("../twitter_tokens.csv")
twitter_tokens.keys()


class TwitterAPI(object):
    """
        TwitterAPI class allows the Connection to Twitter via OAuth
        once you have registered with Twitter and receive the
        necessary credentials.
    """
    # Initialize key variables and get the twitter credentials
    def __init__(self):
        consumer_key = twitter_tokens.values.flatten()[0]
        consumer_secret = twitter_tokens.values.flatten()[1]
        access_token = twitter_tokens.values.flatten()[2]
        access_secret = twitter_tokens.values.flatten()[3]
        
        self.consumer_key = consumer_key
        self.consumer_secret = consumer_secret
        self.access_token = access_token
        self.access_secret = access_secret
        
    # Authenticate credentials with Twitter using OAuth
        self.auth = twitter.oauth.OAuth(access_token, access_secret, 
                                        consumer_key, consumer_secret)
        
        
    # Create registered Twitter API
        self.api = twitter.Twitter(auth=self.auth)
        
        
    # Search Twitter with query q (i.e "ApacheSpark") and max result
    def searchTwitter(self, q, max_res=10, **kwargs):
        search_results = self.api.search.tweets(q=q, count=10, **kwargs)
        statuses = search_results['statuses']
        max_results = min(1000, max_res)
        
        for _ in range(10):
            try:
                next_results = search_results['search_metadata']['next_results']
            except KeyError as e:
                break
            
            next_results = urlparse.parse_qsl(next_results[1:])
            kwargs = dict(next_results)
            
            search_results = self.api.search.tweets(**kwargs)
            statuses += search_results['statuses']
            
            if len(statuses) > max_results:
                break
            
        return statuses
    
    
    
    # Parse tweets as it is collected to extract ID, creation date, userID, tweet text
    def parseTweets(self, statuses):
        tweetx = [(status['id'],
                   status['created_at'],
                   status['user']['id'],
                   status['user']['name'],
                   url['expanded_url'],
                   status['text']) 
                    for status in statuses 
                      for url in status['entities']['urls']
                 ]
        return tweetx
    

# Instantiate the class with the required authentication
obj = TwitterAPI()

# Run a query on the search tern
twtx = obj.searchTwitter("ApacheSpark")

# Parse the tweets
parsed_tweetx = obj.parseTweets(twtx)

# Display output of parsed tweets
print("Lenth of parsed tweets: {} \n\n".format(len(parsed_tweetx)))

# Serialize the data into CSV
csv_fields = ['id', 'created_at', 'user_id', 'user_name', 'tweet_text', 'url']
tweets_2frames = pd.DataFrame(parsed_tweetx, columns=csv_fields)
tweets_2frames.to_csv("tweets.csv", encoding='utf-8')

# Display first 3 rows
tweets_2frames.ix[:2]


tweets_2frames.url[2]



