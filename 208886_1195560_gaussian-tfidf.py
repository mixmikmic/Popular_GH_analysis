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

class GaussianNB:
    def __init__(self, epsilon):
        self.EPSILON = epsilon
        pass

    def fit(self, X, y):
        separated = [[x for x, t in zip(X, y) if t == c] for c in np.unique(y)]
        self.model = np.array([np.c_[np.mean(i, axis=0), np.std(i, axis=0)]
                    for i in separated])

    def _prob(self, x, mean, std):
        exponent = np.exp(- ((x - mean)**2 / ((2 * std**2)+self.EPSILON)))
        return np.log((exponent / ((np.sqrt(2 * np.pi) * std)+self.EPSILON)))

    def predict_log_proba(self, X):
        return [[sum(self._prob(i, *s) for s, i in zip(summaries, x))
                for summaries in self.model] for x in X]

    def predict(self, X):
        return np.argmax(self.predict_log_proba(X), axis=1)

    def score(self, X, y):
        return sum(self.predict(X) == y) / len(y)

gaussian_bayes = GaussianNB(1e-8)
gaussian_bayes.fit(train_X, train_Y)

gaussian_bayes.score(train_X, train_Y)

gaussian_bayes.score(test_X, test_Y)

