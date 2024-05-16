from __future__ import print_function
from time import time
from random import randint

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.cross_validation import train_test_split

import numpy as np
import os
import pickle

validDocsDict = dict()
fileList = os.listdir("BioMedProcessed")
for f in fileList:
    validDocsDict.update(pickle.load(open("BioMedProcessed/" + f, "rb")))

n_samples = len(validDocsDict.keys())
n_features = 10000
n_topics = 2
n_top_words = 30
lengthOfIntroToAdd = 700

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
    elif k.startswith("introduction") and len(validDocsDict[k]) > 10000:
        introCount += 1
        introductionSections.append(validDocsDict[k])

print(len(documents))
print(concLengthTotal * 1.0/ concCount)
print(discLengthTotal * 1.0/ discCount)
print(introCount)

validDocs2 = []
labels2 = []
fileList = os.listdir("PubMedProcessed")
for f in fileList[0:len(fileList)/3]:
    tempDict = pickle.load(open("PubMedProcessed/" + f, "rb"))
    for item in tempDict.keys():
        if item.startswith("conclusion"):
            labels2.append("conclusion")
            validDocs2.append(tempDict[item])
        elif item.startswith("discussion"):
            labels2.append("discussion")
            validDocs2.append(tempDict[item])
        elif item.startswith("introduction") and len(tempDict[item]) > 10000:
            introCount += 1
            introductionSections.append(tempDict[item])

print(len(validDocs2))
print(introCount)

for item in range(len(documents)):
    intro = introductionSections[randint(0, len(introductionSections) - 1)].split(" ")
    randNum = randint(0, len(intro) - lengthOfIntroToAdd)
    introWords = intro[randNum:randNum + lengthOfIntroToAdd]
    documents[item] = documents[item] + " ".join(introWords)

for item in range(len(validDocs2)):
    intro = introductionSections[randint(0, len(introductionSections) - 1)].split(" ")
    randNum = randint(0, len(intro) - lengthOfIntroToAdd)
    introWords = intro[randNum:randNum + lengthOfIntroToAdd]
    validDocs2[item] = validDocs2[item] + " ".join(introWords)
    
train, test, labelsTrain, labelsTest = train_test_split(documents, labels, test_size = 0.1)

# Use tf (raw term count) features for LDA.
print("Extracting tf features for LDA...")
tf_vectorizer = TfidfVectorizer(max_df=0.95, norm = 'l1', min_df=2, max_features=n_features, stop_words='english')
t0 = time()
tf_vectorizer.fit(train)
tf = tf_vectorizer.transform(train)

tfTest = tf_vectorizer.transform(test)
test = tfTest
train = tf

pubTest = tf_vectorizer.transform(validDocs2)

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

results = lda.transform(pubTest)
totalConTop1 = 0
totalConTop2 = 0
totalDisTop1 = 0
totalDisTop2 = 0
for x in range(len(results)):
    val1 = results[x][0]
    val2 = results[x][1]
    total = val1 + val2
    if val1 > val2:
        if labels2[x] == "conclusion":
            totalConTop1 += 1
        else:
            totalDisTop1 += 1
    else:
        if labels2[x] == "conclusion":
            totalConTop2 += 1
        else:
            totalDisTop2 += 1
print("Total Conclusion Topic One: " + str(totalConTop1))
print("Total Conclusion Topic Two: " + str(totalConTop2))
print("Total Discussion Topic One: " + str(totalDisTop1))
print("Total Discussion Topic Two: " + str(totalDisTop2))

from sklearn.naive_bayes import BernoulliNB

classifier = BernoulliNB()

classifier.fit(train.toarray(), labelsTrain)

classResults = classifier.predict(pubTest.toarray())

numRight = 0

for item in range(len(classResults)):
    if classResults[item] == labels2[item]:
        numRight += 1

print(str(numRight * 1.0 / len(classResults) * 1.0))

from sklearn.naive_bayes import BernoulliNB

classifier = BernoulliNB()

classifier.fit(pubTest.toarray(), labels2)

classResults = classifier.predict(train.toarray())

numRight = 0

for item in range(len(classResults)):
    if classResults[item] == labelsTrain[item]:
        numRight += 1

print(str(numRight * 1.0 / len(classResults) * 1.0))

probas = classifier.predict_log_proba(train.toarray())

TotalRight = 0
TotalWrong = 0
numRight = 0
numWrong = 0
RightNumbers = []
WrongNumbers = []
for item in range(len(classResults)):
    if classResults[item] == labelsTrain[item]:
        TotalRight += probas[item][0] + probas[item][1]
        numRight += 1
        RightNumbers.append(probas[item][0] + probas[item][1])
    else:
        TotalWrong += probas[item][0] + probas[item][1]
        numWrong += 1
        WrongNumbers.append(probas[item][0] + probas[item][1])

print(str(TotalRight * 1.0 / numRight))
print(str(TotalWrong * 1.0 / numWrong))

from sklearn.tree import DecisionTreeClassifier

classifier = DecisionTreeClassifier()

classifier.fit(train.toarray(), labelsTrain)

classResults = classifier.predict(pubTest.toarray())

numRight = 0

for item in range(len(classResults)):
    if classResults[item] == labels2[item]:
        numRight += 1

print(str(numRight * 1.0 / len(classResults) * 1.0))

from sklearn.tree import DecisionTreeClassifier

classifier = DecisionTreeClassifier()

classifier.fit(pubTest.toarray(), labels2)

classResults = classifier.predict(train.toarray())

numRight = 0

for item in range(len(classResults)):
    if classResults[item] == labelsTrain[item]:
        numRight += 1

print(str(numRight * 1.0 / len(classResults) * 1.0))

probas = classifier.predict_log_proba(train.toarray())

TotalRight = 0
TotalWrong = 0
numRight = 0
numWrong = 0
RightNumbers = []
WrongNumbers = []
for item in range(len(classResults)):
    if classResults[item] == labelsTrain[item]:
        TotalRight += probas[item][0] + probas[item][1]
        numRight += 1
        RightNumbers.append(probas[item][0] + probas[item][1])
    else:
        TotalWrong += probas[item][0] + probas[item][1]
        numWrong += 1
        WrongNumbers.append(probas[item][0] + probas[item][1])

print(str(TotalRight * 1.0 / numRight))
print(str(TotalWrong * 1.0 / numWrong))



