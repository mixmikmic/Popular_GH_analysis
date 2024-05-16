import nltk

nltk.corpus.gutenberg.fileids()

md = nltk.corpus.gutenberg.words("melville-moby_dick.txt")

md[:8]

md.count("whale")

md.count("boat")

md.count("Ahab")

md.count("laptop")

len(md)

md_set = set(md)

len(md_set)

from __future__ import division #we import this since we are using Python 2.7

len(md)/len(md_set)

md_sents = nltk.corpus.gutenberg.sents("melville-moby_dick.txt")

len(md)/len(md_sents)

