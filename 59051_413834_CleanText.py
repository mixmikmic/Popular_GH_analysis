# Here are the imports we'll use
import pandas as pd
import re
import nltk
from nltk.stem.snowball import SnowballStemmer

# Let's grab all the tweets from https://data.world/data4democracy/far-right/file/sample_tweets.json. Here is the URL:
df = pd.read_json('https://query.data.world/s/bsbt4eb4g8sm4dsgi7w2ecbkt')

# Let's take a look at it
df.head()
# Does anyone know the difference between message and text?

print('Now we have all the tweets inside a {}'.format(type(df)))
print('There are a total of {} tweets in our dataset'.format(len(df)))

# Let's see what different columns we have
print('Here are the columns we have: \n   {}'.format(df.columns))

# Let's start by tokenizing all the words
df['tokenized'] = df['text'].apply (lambda row: nltk.word_tokenize(row))

# Let's add part of speech tags. This function can take a bit of time if it's a large dataset
df['tags']=df['tokenized'].apply(lambda row: nltk.pos_tag(row))

# Now let's remove stop words (e.g. and, to, an, etc.)
# We'll build a little function for that
def remove_stop_words(text):
    filtered = [word for word in text if word not in nltk.corpus.stopwords.words('english')]
    return filtered

df['no_stop'] = df['tokenized'].apply(lambda row: remove_stop_words(row))

# Now we can stem the remaining words
stemmer = SnowballStemmer("english")
df['stems'] = df['no_stop'].apply(lambda words: 
                                    [stemmer.stem(word) for word in words])

# OK, let's take another look at the dataframe
df.head()

# Let's see what variables we have so we know what to store for other datasets
get_ipython().magic('who')

# Since we add everything to the dataframe that's the only thing that appears to be worth storing
get_ipython().magic('store df')

