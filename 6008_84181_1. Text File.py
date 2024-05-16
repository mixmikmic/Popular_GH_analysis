import nltk

di = open("dec_independence.txt")

di_text = di.read()

di_text

type(di_text)

nltk.word_tokenize(di_text)

di_token = nltk.word_tokenize(di_text)

nltk.FreqDist(di_token)

