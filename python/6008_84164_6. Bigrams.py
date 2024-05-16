import nltk

text = "I think it might rain today."

tokens = nltk.word_tokenize(text)

tokens

bigrams =  nltk.bigrams(tokens)

for item in bigrams:
    print item

trigrams = nltk.trigrams(tokens)

for item in trigrams:
    print item

from nltk.util import ngrams

text = "If it is nice out, I will go to the beach."

tokens = nltk.word_tokenize(text)

bigrams = ngrams(tokens,2)

for item in bigrams:
    print item

fourgrams = ngrams(tokens,4)

for item in fourgrams:
    print item

def n_grams(text,n):
    tokens = nltk.word_tokenize(text)
    grams = ngrams(tokens,n)
    return grams

text = "I think it might rain today, but if it is nice out, I will go to the beach."

grams = n_grams(text, 5)

for item in grams:
    print item

