import gensim
import pymongo
import json
import numpy as np
import pandas as pd
from pymongo import MongoClient

import requests

from gensim import corpora, models, similarities

mongoClient = MongoClient()
db = mongoClient.data4democracy
tweets_collection = db.tweets

from gensim.models.word2vec import Word2Vec
from gensim.parsing.preprocessing import STOPWORDS
from gensim.utils import smart_open, simple_preprocess
def tokenize(text):
    return [token for token in simple_preprocess(text) if token not in STOPWORDS]

tweets_model = Word2Vec.load_word2vec_format('../../../../Volumes/SDExternal2/word2vec_twitter_model/word2vec_twitter_model.bin', binary=True, unicode_errors='ignore')

#now calculate word simiarities on twitter data e.g.  
tweets_model.most_similar('jewish')

#to remind myself what a tweet is like:
r = requests.get('https://s3-us-west-2.amazonaws.com/discursive/2017/1/10/18/tweets-25.json')

tweets_collection = r.json()
print(tweets_collection[0])
#for text analysis, the 'text' field is the one of interest

#the tweets text are in the 'text' field
print(tweets_collection[0]['text'])

tweets_text_documents = [x['text'] for x in tweets_collection]

#quick check that the mapping was done correctly
tweets_text_documents[0]

#quick check of the tokenize function -- remove stopwords included 
tokenize(tweets_text_documents[0])

tokenized_tweets = [[word for word in tokenize(x) if word != 'rt'] for x in tweets_text_documents]

tokenized_tweets[0]

#construct a dictoinary of the words in the tweets using gensim
# the dictionary is a mapping between words and their ids
tweets_dictionary = corpora.Dictionary(tokenized_tweets)

#save gyhe dict for future reference
tweets_dictionary.save('temp/tweets_dictionary.dict')

#just a quick view of words and ids
dict(list(tweets_dictionary.token2id.items())[0:20])

#convert tokenized documents to vectors
# compile corpus (vectors number of times each elements appears)
tweet_corpus = [tweets_dictionary.doc2bow(x) for x in tokenized_tweets]
corpora.MmCorpus.serialize('temp/tweets_corpus.mm', tweet_corpus) # save for future ref

tweets_tfidf_model = gensim.models.TfidfModel(tweet_corpus, id2word = tweets_dictionary)

tweets_tfidf_model[tweet_corpus]

#Create similarity matrix of all tweets
'''note from gensim docs: The class similarities.MatrixSimilarity is only appropriate when 
   the whole set of vectors fits into memory. For example, a corpus of one million documents 
   would require 2GB of RAM in a 256-dimensional LSI space, when used with this class.
   Without 2GB of free RAM, you would need to use the similarities.Similarity class.
   This class operates in fixed memory, by splitting the index across multiple files on disk, 
   called shards. It uses similarities.MatrixSimilarity and similarities.SparseMatrixSimilarity internally,
   so it is still fast, although slightly more complex.'''
index = similarities.MatrixSimilarity(tweets_tfidf_model[tweet_corpus]) 
index.save('temp/tweetsSimilarity.index')

#get similarity matrix between docs: https://groups.google.com/forum/#!topic/gensim/itYEaOYnlEA
#and check that the similarity matrix is what you expect
tweets_similarity_matrix = np.array(index)
print(tweets_similarity_matrix.shape)

#save the similarity matrix and associated tweets to json
#work in progress-- use tSNE to visualise the tweets to see if there's any clustering
outputDict = {'tweets' : [{'text': x['text'], 'id': x['id_str'], 'user': x['original_name']} for x in tweets_collection], 'matrix': tweets_similarity_matrix.tolist()}
with open('temp/tweetSimilarity.json', 'w') as f:
    json.dump(outputDict, f)

#back to the word2vec idea, use min_count=1 since corpus is tiny
tweets_collected_model = gensim.models.Word2Vec(tokenized_tweets, min_count=1)

#looking again at the term jewish in our small tweet collection...
tweets_collected_model.most_similar('jewish')

