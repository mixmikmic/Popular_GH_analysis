import pandas as pd
import json

# Load the first 10 reviews
f = open('data/yelp/v6/yelp_dataset_challenge_academic_dataset/yelp_academic_dataset_review.json')
js = []
for i in range(10):
    js.append(json.loads(f.readline()))
f.close()
review_df = pd.DataFrame(js)
review_df.shape

import spacy

# model meta data
spacy.info('en')

# preload the language model
nlp = spacy.load('en')

# Keeping it in a pandas dataframe
doc_df = review_df['text'].apply(nlp)

type(doc_df)

type(doc_df[0])

doc_df[4]

# spacy gives you both fine grained (.pos_) + coarse grained (.tag_) parts of speech    
for doc in doc_df[4]:
    print(doc.text, doc.pos_, doc.tag_)

# spaCy also does noun chunking for us

print([chunk for chunk in doc_df[4].noun_chunks])

from textblob import TextBlob

blob_df = review_df['text'].apply(TextBlob)

type(blob_df)

type(blob_df[4])

blob_df[4].tags

# blobs in TextBlob also have noun phrase extraction

print([np for np in blob_df[4].noun_phrases])

