import nltk
import random

from nltk.corpus import names

names.fileids()

import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')
get_ipython().magic('matplotlib inline')

cfd = nltk.ConditionalFreqDist((fileid,name[-2:]) for fileid in names.fileids() for name in names.words(fileid))

plt.figure(figsize=(50,10))
cfd.plot()

def name_feature(name):
    return {'pair': name[-2:]}

name_feature("Katy")

name_list = ([(name, 'male') for name in names.words('male.txt')] + [(name, "female") for name in names.words('female.txt')])

name_list[:10]

name_list[-10:]

random.shuffle(name_list)

name_list[:10]

features = [(name_feature(name), gender) for (name,gender) in name_list]

features[:10]

len(features)/2

training_set = features[:3972]
testing_set = features[3972:]

classifier = nltk.NaiveBayesClassifier.train(training_set)

male_names = names.words('male.txt')
"Carmello" in male_names

classifier.classify(name_feature("Carmello"))

nltk.classify.accuracy(classifier, testing_set)

