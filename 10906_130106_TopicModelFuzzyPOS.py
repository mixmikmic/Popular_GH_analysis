from __future__ import print_function
from time import time
import os
from random import randint

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.cross_validation import train_test_split

import numpy as np

import pickle

validDocsDict = dict()
fileList = os.listdir("BioMedPOS")
for f in fileList:
    validDocsDict.update(pickle.load(open("BioMedPOS/" + f, "rb")))

n_samples = len(validDocsDict.keys())
n_features = 5000
n_topics = 2
n_top_words = 30
lengthOfIntroToAdd = 500

def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))

print("Loading dataset...")
t0 = time()
documents = []
introductionSections = []

labels = []
concLengthTotal = 0
discLengthTotal = 0
concCount = 0
discCount = 0
introCount = 0

for k in validDocsDict.keys():
    if k.startswith("conclusion"):
        labels.append("conclusion")
        documents.append(validDocsDict[k])
        concCount += 1
        concLengthTotal += len(validDocsDict[k].split(' '))
    elif k.startswith("discussion"):
        labels.append("discussion")
        documents.append(validDocsDict[k])
        discCount += 1
        discLengthTotal += len(validDocsDict[k].split(' '))
    elif k.startswith("introduction") and len(validDocsDict[k]) > 5000:
        introCount += 1
        introductionSections.append(validDocsDict[k])

print(len(documents))
print(concLengthTotal * 1.0/ concCount)
print(discLengthTotal * 1.0/ discCount)
print(introCount)

for item in range(len(documents)):
    if labels[item] == "conclusion":
        documents[item] = documents[item] + documents[item]
    #intro = introductionSections[randint(0, len(introductionSections) - 1)].split(" ")
    #randNum = randint(0, len(intro) - lengthOfIntroToAdd)
    #introWords = intro[randNum:randNum + lengthOfIntroToAdd]
    #documents[item] = documents[item] + " ".join(introWords)

train, test, labelsTrain, labelsTest = train_test_split(documents, labels, test_size = 0.5)

trainSetOne = []
trainSetTwo = []

for x in range(len(train)):
    if labelsTrain[x] == "conclusion":
        trainSetOne.append(train[x])
    else:
        trainSetTwo.append(train[x])

# Use tf (raw term count) features for LDA.
print("Extracting tf features for LDA...")
tf_vectorizer = TfidfVectorizer(max_df=0.95, norm = 'l1', min_df=2, max_features=n_features, ngram_range = (1,6))
t0 = time()
tf = tf_vectorizer.fit_transform(train)

tfSetOne = tf_vectorizer.transform(trainSetOne)
tfSetTwo = tf_vectorizer.transform(trainSetTwo)
tfTest = tf_vectorizer.transform(test)
test = tfTest
train = tf
trainSetOne = tfSetOne
trainSetTwo = tfSetTwo

print("done in %0.3fs." % (time() - t0))

print("Fitting LDA models with tf features, n_samples=%d and n_features=%d..."
      % (n_samples, n_features))
lda = LatentDirichletAllocation(n_topics=n_topics, max_iter=100,
                                learning_method='online', learning_offset=50.,
                                random_state=0)

t0 = time()
lda.fit(tf)
print("done in %0.3fs." % (time() - t0))

print("\nTopics in LDA model:")
tf_feature_names = tf_vectorizer.get_feature_names()
print_top_words(lda, tf_feature_names, n_top_words)

results = lda.transform(test)
totalConTop1 = 0
totalConTop2 = 0
totalDisTop1 = 0
totalDisTop2 = 0
for x in range(len(results)):
    val1 = results[x][0]
    val2 = results[x][1]
    total = val1 + val2
    print(str(labelsTest[x]) + " " + str(val1/total) + " " + str(val2/total))
    if val1 > val2:
        if labelsTest[x] == "conclusion":
            totalConTop1 += 1
        else:
            totalDisTop1 += 1
    else:
        if labelsTest[x] == "conclusion":
            totalConTop2 += 1
        else:
            totalDisTop2 += 1

print("Total Conclusion Topic One: " + str(totalConTop1))
print("Total Conclusion Topic Two: " + str(totalConTop2))
print("Total Discussion Topic One: " + str(totalDisTop1))
print("Total Discussion Topic Two: " + str(totalDisTop2))

lda.get_params()

from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()

classifier.fit(train.toarray(), labelsTrain)

classResults = classifier.predict(test.toarray())

numRight = 0

for item in range(len(classResults)):
    if classResults[item] == labelsTest[item]:
        numRight += 1

print(str(numRight * 1.0 / len(classResults) * 1.0))

from sklearn.naive_bayes import BernoulliNB

classifier = BernoulliNB()

classifier.fit(train.toarray(), labelsTrain)

classResults = classifier.predict(test.toarray())

numRight = 0

for item in range(len(classResults)):
    if classResults[item] == labelsTest[item]:
        numRight += 1

print(str(numRight * 1.0 / len(classResults) * 1.0))

from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier()

classifier.fit(train, labelsTrain)

classResults = classifier.predict(test)
numRight = 0

for item in range(len(classResults)):
    if classResults[item] == labelsTest[item]:
        numRight += 1

print(str(numRight * 1.0 / len(classResults) * 1.0))

from sklearn.tree import DecisionTreeClassifier

classifier = DecisionTreeClassifier()

classifier.fit(train.toarray(), labelsTrain)

classResults = classifier.predict(test.toarray())
numRight = 0
numWrongDisc = 0
numWrongConc = 0

for item in range(len(classResults)):
    if classResults[item] == labelsTest[item]:
        numRight += 1
    else:
        if classResults[item] == "discussion":
            numWrongDisc += 1
        else:
            numWrongConc += 1

print(str(numRight * 1.0 / len(classResults) * 1.0))
print("Incorrectly classified as discussion: " + str(numWrongDisc))
print("Incorrectly classified as conclusion: " + str(numWrongConc))
print(len(classResults))

ldaSet1 = LatentDirichletAllocation(n_topics=20, max_iter=100,
                                learning_method='online', learning_offset=50.,
                                random_state=0)
ldaSet2 = LatentDirichletAllocation(n_topics=20, max_iter=100,
                                learning_method='online', learning_offset=50.,
                                random_state=0)

ldaSet1.fit(trainSetOne)
print_top_words(ldaSet1, tf_feature_names, n_top_words)

ldaSet2.fit(trainSetTwo)
print_top_words(ldaSet2, tf_feature_names, n_top_words)

results1 = ldaSet1.transform(train)
results2 = ldaSet2.transform(train)

resultsTest1 = ldaSet1.transform(test)
resultsTest2 = ldaSet2.transform(test)

results = np.hstack((results1, results2))
resultsTest = np.hstack((resultsTest1, resultsTest2))

from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()

classifier.fit(results, labelsTrain)

classResults = classifier.predict(resultsTest)

numRight = 0

for item in range(len(classResults)):
    if classResults[item] == labelsTest[item]:
        numRight += 1

print(str(numRight * 1.0 / len(classResults) * 1.0))

from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier()

classifier.fit(results, labelsTrain)

classResults = classifier.predict(resultsTest)

numRight = 0

for item in range(len(classResults)):
    if classResults[item] == labelsTest[item]:
        numRight += 1

print(str(numRight * 1.0 / len(classResults) * 1.0))

for x in range(len(results)):
    total = 0
    for y in range(len(results[x])):
        total += results[x][y]
    for y in range(len(results[x])):
        results[x][y] = results[x][y]/total
        
for x in range(len(resultsTest)):
    total = 0
    for y in range(len(resultsTest[x])):
        total += resultsTest[x][y]
    for y in range(len(resultsTest[x])):
        resultsTest[x][y] = resultsTest[x][y]/total

from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()

classifier.fit(results, labelsTrain)

classResults = classifier.predict(resultsTest)

numRight = 0

for item in range(len(classResults)):
    if classResults[item] == labelsTest[item]:
        numRight += 1

print(str(numRight * 1.0 / len(classResults) * 1.0))

from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier()

classifier.fit(results, labelsTrain)

classResults = classifier.predict(resultsTest)

numRight = 0

for item in range(len(classResults)):
    if classResults[item] == labelsTest[item]:
        numRight += 1

print(str(numRight * 1.0 / len(classResults) * 1.0))



