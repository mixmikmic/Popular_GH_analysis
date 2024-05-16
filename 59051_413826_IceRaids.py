import pandas as pd
import re
import math
from nltk.classify import NaiveBayesClassifier
import collections
import nltk

# Let's pull in the tweets from the extracted ICE tweets
df = pd.read_csv('https://s3.amazonaws.com/d4d-public/public/ice_extract.csv')

# Let's take a look at it
df.head()

# OK, now that we have that, let's save it to a text file. This will help get a quick way to look at the text
df['text'].to_csv('training_data/test.txt', index=False)

#I have two files, one of positive statements (from movie reviews) and the other with negative. To run this
#download the files from Andy Bromberg's GitHub page: https://github.com/abromberg/sentiment_analysis_python
positive_statements = 'C:/Users/HMGSYS/Google Drive/JupyterNotebooks/Data4Democracy/training_data/pos.txt'
negative_statements = 'C:/Users/HMGSYS/Google Drive/JupyterNotebooks/Data4Democracy/training_data/neg.txt'
#And I also have our file we just created with ICERaids tweets
test_statements = 'C:/Users/HMGSYS/Google Drive/JupyterNotebooks/Data4Democracy/training_data/test.txt'

#creates a feature selection mechanism that uses all words
def make_full_dict(words):
    return dict([(word, True) for word in words])

# Let's open the files and create lists with all the words in them
posFeatures = []
negFeatures = []
mytestFeatures = []
with open(positive_statements, 'r') as posSentences:
    for i in posSentences:
        posWords = re.findall(r"[\w']+|[.,!?;]", i.rstrip())
        posWords = [make_full_dict(posWords), 'pos']
        posFeatures.append(posWords)
with open(negative_statements, 'r') as negSentences:
    for i in negSentences:
        negWords = re.findall(r"[\w']+|[.,!?;]", i.rstrip())
        negWords = [make_full_dict(negWords), 'neg']
        negFeatures.append(negWords)
# Now let's do the same with our test data
with open(test_statements, 'r') as mytestSentences:
    for i in mytestSentences:
        mytestWords = re.findall(r"[\w']+|[.,!?;]", i.rstrip())
        # We're going to label them as positive so we can check the accuracy
        mytestWords = [make_full_dict(mytestWords), 'pos']
        mytestFeatures.append(mytestWords)

# Let's take a quick look at our result
posFeatures[0:2]

#selects 3/4 of the features to be used for training and 1/4 to be used for testing
posCutoff = int(math.floor(len(posFeatures)*3/4))
negCutoff = int(math.floor(len(negFeatures)*3/4))
mytestCutoff = int(math.floor(len(mytestFeatures)*3/4))
#Now this is a bit tricky because we have testFeatures and mytestFeatures. testFeatures is from the Bromberg model
#mytestFeatures is me throwing our test (ICERaids) tweets into the same process
trainFeatures = posFeatures[:posCutoff] + negFeatures[:negCutoff]
testFeatures = posFeatures[posCutoff:] + negFeatures[negCutoff:]
#This last one doesn't change
mytestFeatures = mytestFeatures

# We'll start with a Naive Bayes Classifier. There's a lot more we could do here but it's a start
classifier = NaiveBayesClassifier.train(trainFeatures)

#initiates referenceSets and testSets
referenceSets = collections.defaultdict(set)
testSets = collections.defaultdict(set)

# puts correctly labeled sentences in referenceSets and the predictively labeled version in testsets
for i, (features, label) in enumerate(testFeatures):
    referenceSets[label].add(i)
    predicted = classifier.classify(features)
    testSets[predicted].add(i)

#prints metrics to show how well the feature selection did
print ('train on %d instances, test on %d instances' % (len(trainFeatures), len(testFeatures)))
print ('accuracy:', nltk.classify.util.accuracy(classifier, testFeatures))
print ('pos precision:', nltk.scores.precision(referenceSets['pos'], testSets['pos']))
print ('pos recall:', nltk.scores.recall(referenceSets['pos'], testSets['pos']))
print ('neg precision:', nltk.scores.precision(referenceSets['neg'], testSets['neg']))
print ('neg recall:', nltk.scores.recall(referenceSets['neg'], testSets['neg']))

classifier.show_most_informative_features(10)

print ('This model predicts that {:.1%} of tweets about the ICE raids have been positive'
       .format(nltk.classify.util.accuracy(classifier, mytestFeatures)))



