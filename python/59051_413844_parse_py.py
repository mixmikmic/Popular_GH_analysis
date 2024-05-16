import gensim
import os
import numpy as np
import itertools
import json
import re
import pymoji
import importlib
from nltk.tokenize import TweetTokenizer
from gensim import corpora
import string
from nltk.corpus import stopwords
from six import iteritems
import csv

tokenizer = TweetTokenizer()

def keep_retweets(tweets_objs_arr):
    return [x["text"] for x in tweets_objs_arr if x['retweet'] != 'N'], [x["name"] for x in tweets_objs_arr if x['retweet'] != 'N'], [x["followers"] for x in tweets_objs_arr if x['retweet'] != 'N']

def convert_emojis(tweets_arr):
    return [pymoji.replaceEmojiAlt(x, trailingSpaces=1) for x in tweets_arr]

def tokenize_tweets(tweets_arr):
    result = []
    for x in tweets_arr:
        try:
            tokenized = tokenizer.tokenize(x)
            result.append([x.lower() for x in tokenized if x not in string.punctuation])
        except:
            pass
#             print(x)
    return result

class Tweets(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for root, directories, filenames in os.walk(self.dirname):
            for filename in filenames:
                if(filename.endswith('json')):
                    print(root + filename)
                    with open(os.path.join(root,filename), 'r') as f:
                        data = json.load(f)
                        data_parsed_step1, user_names, followers = keep_retweets(data)
                        data_parsed_step2 = convert_emojis(data_parsed_step1)
                        data_parsed_step3 = tokenize_tweets(data_parsed_step2)
                        for data, name, follower in zip(data_parsed_step3, user_names, followers):
                            yield name, data, follower


#model = gensim.models.Word2Vec(sentences, workers=2, window=5, sg = 1, size = 100, max_vocab_size = 2 * 10000000)
#model.save('tweets_word2vec_2017_1_size100_window5')
#print('done')
#print(time.time() - start_time)

# building the dictionary first, from the iterator
sentences = Tweets('/media/henripal/hd1/data/2017/1/') # a memory-friendly iterator
dictionary = corpora.Dictionary((tweet for _, tweet, _ in sentences))

# here we use the downloaded  stopwords from nltk and create the list
# of stop ids using the hash defined above
stop = set(stopwords.words('english'))
stop_ids = [dictionary.token2id[stopword] for stopword in stop if stopword in dictionary.token2id]

# and this is the items we don't want - that appear less than 20 times
# hardcoded numbers FTW
low_freq_ids = [tokenid for tokenid, docfreq in iteritems(dictionary.dfs) if docfreq  <1500]

# finally we filter the dictionary and compactify
dictionary.filter_tokens(stop_ids + low_freq_ids)
dictionary.compactify()  # remove gaps in id sequence after words that were removed
print(dictionary)

# reinitializing the iterator to get more stuff
sentences = Tweets('/media/henripal/hd1/data/2017/1/')
corpus = []
name_to_follower = {}
names = []

for name, tweet, follower in sentences:
    corpus.append(tweet) 
    names.append(name)
    name_to_follower[name] = follower

with open('/media/henripal/hd1/data/name_to_follower.csv', 'w') as csv_file:
    writer = csv.writer(csv_file)
    for key, value in name_to_follower.items():
        writer.writerow([key, value])

with open('/media/henripal/hd1/dta/corpus_names.csv', 'w') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(names)

# now we save the sparse bow corpus matrix using matrix market format
corpora.MmCorpus.serialize('/media/henripal/hd1/data/corp.mm', corpus)

# and we save the dictionary as a text file
dictionary.save('/media/henripal/hd1/data/dict')

