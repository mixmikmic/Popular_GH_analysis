import matplotlib.pyplot as plt

# make sure that graphs are embedded into our notebook output
get_ipython().magic('matplotlib inline')

plt.plot() # create an empty graph

import nltk

# read Gold Bug plain text into string
with open("data/goldBug.txt", "r") as f:
    goldBugString = f.read()

# simple lowercase tokenize
goldBugTokensLowercase = nltk.word_tokenize(goldBugString.lower())

# filter out tokens that aren't words
goldBugWordTokensLowercase = [word for word in goldBugTokensLowercase if word[0].isalpha()]

# determine frequencies
goldBugWordTokensLowercaseFreqs = nltk.FreqDist(goldBugWordTokensLowercase)

# preview the top 20 frequencies
goldBugWordTokensLowercaseFreqs.tabulate(20)

# make sure that graphs are embedded into our notebook output
get_ipython().magic('matplotlib inline')

# plot the top frequency words in a graph
goldBugWordTokensLowercaseFreqs.plot(25, title="Top Frequency Word Tokens in Gold Bug")

goldBugLowerCaseWordTokenLengths = [len(w) for w in goldBugWordTokensLowercase]
print("first five words: ", goldBugWordTokensLowercase[:5])
print("first five word lengths: ", goldBugLowerCaseWordTokenLengths[:5])

nltk.FreqDist(goldBugLowerCaseWordTokenLengths).plot()

goldBugLowerCaseWordTokenLengthFreqs = list(nltk.FreqDist(goldBugLowerCaseWordTokenLengths).items())
goldBugLowerCaseWordTokenLengthFreqs # sorted by word length (not frequency)

import matplotlib.pyplot as plt

goldBugLowerCaseWordTokenWordLengths = [f[0] for f in goldBugLowerCaseWordTokenLengthFreqs]
goldBugLowerCaseWordTokenWordLengthValues = [f[1] for f in goldBugLowerCaseWordTokenLengthFreqs]
plt.plot(goldBugLowerCaseWordTokenWordLengths, goldBugLowerCaseWordTokenWordLengthValues)
plt.xlabel('Word Length')
plt.ylabel('Word Count')

stopwords = nltk.corpus.stopwords.words("English")
goldBugContentWordTokensLowercase = [word for word in goldBugWordTokensLowercase if word not in stopwords]
goldBugContentWordTokensLowercaseFreqs = nltk.FreqDist(goldBugContentWordTokensLowercase)
goldBugContentWordTokensLowercaseFreqs.most_common(20)

goldBugContentWordTokensLowercaseFreqs.plot(20, title="Top Frequency Content Terms in Gold Bug")

goldBugText = nltk.Text(nltk.word_tokenize(goldBugString))
goldBugText.concordance("jupiter", lines=5)
goldBugText.concordance("legrand", lines=5)

goldBugText.dispersion_plot(["Jupiter", "Legrand"])

