import boto3
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt

session = boto3.Session(profile_name='default')
s3 = session.resource('s3')
bucket = s3.Bucket("far-right")
session.available_profiles

# print all objects in bucket
for obj in bucket.objects.all():
    if "chan" in obj.key:
        #print(obj.key)
        pass

bucket.download_file('fourchan/chan_example.csv', 'chan_example.csv')

chan = pd.read_csv("chan_example.csv")
# remove the newline tags.  They're not useful for our analysis and just clutter the text.
chan.com = chan.com.astype(str).apply(lambda x: x.replace("<br>", " "))

bucket.download_file('info-source/daily/20170228/fourchan/fourchan_1204.json', '2017-02-28-1204.json')
chan2 = pd.read_json("2017-02-28-1204.json")

soup = BeautifulSoup(chan.com[19], "lxml")
quotes = soup.find("span")
for quote in quotes.contents:
    print(quote.replace(">>", ""))
parent = soup.find("a")
print(parent.contents[0].replace(">>", ""))

print(chan.com[19])

# If there's a quote and then the text, this would work. 
print(chan.com[19].split("</span>")[-1])

def split_comment(comment):
    """Splits up a comment into parent, quotes, and text"""
    
    # I used lxml to 
    soup = BeautifulSoup(comment, "lxml")
    quotes, quotelink, text = None, None, None
    try:
        quotes = soup.find("span")
        quotes = [quote.replace(">>", "") for quote in quotes.contents]
    except:
        pass
    try:
        quotelink = soup.find("a").contents[0].replace(">>", "")
    except: 
        pass
    # no quote or parent
    if quotes is None and quotelink is None:
        text = comment
    # Parent but no quote
    if quotelink is not None and quotes is None:
        text = comment.split("a>")[-1]
    # There is a quote
    if quotes is not None:
        text = comment.split("</span>")[-1]
    return {'quotes':quotes, 'quotelink': quotelink, 'text': text}

df = pd.DataFrame({'quotes':[], 'quotelink':[], 'text':[]})
for comment in chan['com']:
    df = df.append(split_comment(comment), ignore_index = True)
    
full = pd.merge(chan, df, left_index = True, right_index = True)

quotes = pd.Series()
quotelinks = pd.Series()
texts = pd.Series()
for comment in chan['com']:
    parse = split_comment(comment)
    quotes.append(pd.Series(parse['quotes']))
    quotelinks.append(pd.Series(parse['quotelink']))
    texts.append(pd.Series(parse['text']))
chan['quotes'] = quotes
chan['quotelinks'] = quotelinks
chan['text'] = texts

threads = full['parent'].unique()
full_text = {}
for thread in threads:
    full_text[int(thread)] = ". ".join(full[full['parent'] == thread]['text'])

import gensim
import pyLDAvis.gensim as gensimvis
import pyLDAvis

tokenized_messages = []
for msg in nlp.pipe(full['text'], n_threads = 100, batch_size = 100):
    ents = msg.ents
    msg = [token.lemma_ for token in msg if token.is_alpha and not token.is_stop]
    tokenized_messages.append(msg)

# Build the corpus using gensim     
dictionary = gensim.corpora.Dictionary(tokenized_messages)
msg_corpus = [dictionary.doc2bow(x) for x in tokenized_messages]
msg_dictionary = gensim.corpora.Dictionary([])
          
# gensim.corpora.MmCorpus.serialize(tweets_corpus_filepath, tweets_corpus)

import nltk
from nltk.classify import NaiveBayesClassifier
from nltk.classify import accuracy
from nltk import WordNetLemmatizer
lemma = nltk.WordNetLemmatizer()
df = pd.read_csv('https://query.data.world/s/8c7bwy8c55zx1t0c4yyrnjyax')

emotions = list(df.groupby("sentiment").agg("count").sort_values(by = "content", ascending = False).head(6).index)
print(emotions)
emotion_subset = df[df['sentiment'].isin(emotions)]

def format_sentence(sent):
    ex = [i.lower() for i in sent.split()]
    lemmas = [lemma.lemmatize(i) for i in ex]
    
    return {word: True for word in nltk.word_tokenize(" ".join(lemmas))}


def create_train_vector(row):
    """
    Formats a row when used in df.apply to create a train vector to be used by a 
    Naive Bayes Classifier from the nltk library.
    """
    sentiment = row[1]
    text = row[3]
    return [format_sentence(text), sentiment]

train = emotion_subset.apply(create_train_vector, axis = 1)
# Split off 10% of our train vector to be for test.

test = train[:int(0.1*len(train))]
train = train[int(0.9)*len(train):]

emotion_classifier = NaiveBayesClassifier.train(train)

print(accuracy(emotion_classifier, test))

emotion_classifier.show_most_informative_features()

for comment in full['text'].head(10):
    print(emotion_classifier.classify(format_sentence(comment)), ": ", comment)

full['emotion'] = full['text'].apply(lambda x: emotion_classifier.classify(format_sentence(x)))

grouped_emotion_messages = full.groupby('emotion').count()[[2]]
grouped_emotion_messages.columns = ["count"]
grouped_emotion_messages

grouped_emotion_messages.plot.bar()

