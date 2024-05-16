import sklearn.datasets
import re
from sklearn.cross_validation import train_test_split
import numpy as np

def clearstring(string):
    string = re.sub('[^A-Za-z0-9 ]+', '', string)
    string = string.split(' ')
    string = filter(None, string)
    string = [y.strip() for y in string]
    string = ' '.join(string)
    return string

def separate_dataset(trainset):
    datastring = []
    datatarget = []
    for i in range(len(trainset.data)):
        data_ = trainset.data[i].split('\n')
        data_ = list(filter(None, data_))
        for n in range(len(data_)):
            data_[n] = clearstring(data_[n])
        datastring += data_
        for n in range(len(data_)):
            datatarget.append(trainset.target[i])
    return datastring, datatarget

trainset = sklearn.datasets.load_files(container_path = 'local', encoding = 'UTF-8')
trainset.data, trainset.target = separate_dataset(trainset)
print (trainset.target_names)
print (len(trainset.data))
print (len(trainset.target))

vocabulary = list(set(' '.join(trainset.data).split()))
len(vocabulary)

# calculate IDF
idf = {}
for i in vocabulary:
    idf[i] = 0
    for k in trainset.data:
        if i in k.split():
            idf[i] += 1
    idf[i] = np.log(idf[i] / len(trainset.data))

# calculate TF
X = np.zeros((len(trainset.data),len(vocabulary)))
for no, i in enumerate(trainset.data):
    for text in i.split():
        X[no, vocabulary.index(text)] += 1
    for text in i.split():
        # calculate TF * IDF
        X[no, vocabulary.index(text)] = X[no, vocabulary.index(text)] * idf[text]

train_X, test_X, train_Y, test_Y = train_test_split(X, trainset.target, test_size = 0.2)

class MultinomialNB:
    def __init__(self, epsilon):
        self.EPSILON = epsilon

    def fit(self, X, y):
        count_sample = X.shape[0]
        separated = [[x for x, t in zip(X, y) if t == c] for c in np.unique(y)]
        self.class_log_prior_ = [np.log(len(i) / count_sample + self.EPSILON) for i in separated]
        count = np.array([np.array(i).sum(axis=0) for i in separated])
        self.feature_log_prob_ = np.log(count / count.sum(axis=1)[np.newaxis].T + self.EPSILON)

    def predict_log_proba(self, X):
        return [(self.feature_log_prob_ * x).sum(axis=1) + self.class_log_prior_ for x in X]

    def predict(self, X):
        return np.argmax(self.predict_log_proba(X), axis=1)

multinomial_bayes = MultinomialNB(1e-8)
multinomial_bayes.fit(train_X, train_Y)

np.mean(train_Y == multinomial_bayes.predict(train_X))

np.mean(test_Y == multinomial_bayes.predict(test_X))

