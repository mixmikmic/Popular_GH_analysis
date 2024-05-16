import numpy as np
import pymc as pm
#K, V, D = 2, 4, 3 # number of topics, words, documents
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context("poster")
get_ipython().magic('matplotlib inline')

K, V, D = 5, 10, 20 # number of topics, words, documents

#data = np.array([[1, 1, 1, 1], [1, 1, 1, 1], [0, 0, 0, 0]])

data_temp = np.random.randint(0,10,size=(D,V))

class LDA(object):
    
    def __init__(self, data, topics=K, vocab=V):
        """
        Takes the data variable and outputs a model
        """
        self.data = data
        self.topics = topics
        self.vocab = vocab+1
        self.docs = len(self.data)
        self.alpha = np.ones(self.topics)
        self.beta = np.ones(self.vocab)

        self.theta = pm.Container([pm.CompletedDirichlet("theta_%s" % i,                                                         pm.Dirichlet("ptheta_%s" % i, theta=self.alpha))
                                   for i in range(self.docs)])
        self.phi = pm.Container([pm.CompletedDirichlet("phi_%s" % i,                                                       pm.Dirichlet("pphi_%s" % i, theta=self.beta))
                                 for i in range(self.topics)])
        self.Wd = [len(doc) for doc in self.data]

        self.Z = pm.Container([pm.Categorical("z_%s" % d, 
                                              p=self.theta[d],
                                              size=self.Wd[d],
                                              value=np.random.randint(self.topics,size=self.Wd[d]))
                               for d in range(self.docs)])
        self.W = pm.Container([pm.Categorical("w_%s,%s" % (d,i),
                                              p=pm.Lambda("phi_z_%s_%s" % (d,i), 
                                                          lambda z=self.Z[d][i], phi=self.phi: phi[z]),
                                              value=self.data[d][i],
                                              observed=True)
                               for d in range(self.docs) for i in range(self.Wd[d])])

        self.model = pm.Model([self.theta, self.phi, self.Z, self.W])
        self.mcmc = pm.MCMC(self.model)
    
    def fit(self, iterations=1000, burn_in=10):
        # Fit the model by sampling from the data iterations times with burn in of burn_in.
        self.mcmc.sample(iterations, burn=burn_in)
        
    def show_topics(self):
        # Show distribution of topics over words
        return self.phi.value
    
    def show_words(self):
        # Show distribution of words in documents over topics
        return self.W.value
    
    def KLDiv(self, p,q):
        return np.sum(p*np.log10(p/q))
    
    def cosine_sim(self, x,y):
        return np.dot(x,y)/np.sqrt(np.dot(x,x)*np.dot(y,y))
    
    def sorted_docs_sim(self):
        kldivs_docs = [(i, j, self.KLDiv(self.theta[i].value,self.theta[j].value),
                        self.cosine_sim(self.data[i], self.data[j]))
                       for i in range(len(self.theta)) for j in range(len(self.theta))
                       if i != j]
        return sorted(kldivs_docs, key=lambda x: x[3], reverse=True)
    
    def show_topic_words(self, idwords, n=10):
        for i, t in enumerate(self.phi.value):
            print "Topic %i : " % i, ", ".join(idwords[w_] for w_ in np.argsort(t[0])[-10:] if w_ < (self.vocab-1-1))
    
    def plot_data(self):
        plt.clf()
        plt.matshow(data, fignum=1000, cmap=plt.cm.Reds)
        plt.gca().set_aspect('auto')
        plt.xlabel("Words")
        plt.ylabel("Documents")
    
    def plot_words_per_topic(self, ax=None):
        if ax is None:
            plt.clf()
            fig, ax = plt.subplots(1,1)
        words = self.Z.value
        topic_dist = dict()
        for k_i in words:
            for k in k_i:
                if k not in topic_dist:
                    topic_dist[k] = 0
                topic_dist[k] += 1
        ax.bar(topic_dist.keys(), topic_dist.values())
        ax.set_xlabel("Topics")
        ax.set_ylabel("Counts")
        ax.set_title("Document words per topics")
        plt.show()
        
    def plot_word_dist(self, ax=None):
        topics = self.phi.value
        if ax is None:
            plt.clf()
            fig, ax = plt.subplots((len(topics)+1)/2, 2, figsize=(10,10))
        for i, t in enumerate(topics):
            ax[i/2][i%2].bar(range(len(t[0])), t[0])
            ax[i/2][i%2].set_title("Topic %s" % i)
        plt.suptitle("Vocab word proportions per topic")
        fig.subplots_adjust(hspace=0.5, wspace=0.5)

lda = LDA(data_temp)
lda.fit()
lda.plot_words_per_topic()

lda.plot_word_dist()

kld_sorted = lda.sorted_docs_sim()
kld_sorted[:10]

from nltk.corpus import inaugural
from nltk.corpus import stopwords
stops = set(stopwords.words("english"))

# Create a vocabulary from the corpus
vocab = dict()
for fileid in inaugural.fileids():
    for word in inaugural.words(fileid):
        word = word.lower()
        if word not in stops and word.isalpha():
            if word not in vocab:
                vocab[word] = 0
            vocab[word] += 1

"""
Sort the vocab keep only words which occur more than 50 times
Then Create word to id and id to word dictionaries
"""
vocab_sorted = filter(lambda x: x[1] > 50, sorted(vocab.items(), key=lambda x: x[1], reverse=True))
wordids = {v[0]: i for i, v in enumerate(vocab_sorted)}
idwords = {i: v[0] for i, v in enumerate(vocab_sorted)}
vocab_size = len(wordids)
print vocab_size

# Generate corpus document vectors
data = []
for fileid in inaugural.fileids():
    data.append([0]*vocab_size)
    for word in inaugural.words(fileid):
        word = word.lower()
        if word in wordids:
            data[-1][wordids[word]] += 1

len(data)

# Plot the document word matrix
print data[0][:10]
data = np.array(data)
plt.clf()
plt.matshow(data, fignum=1000, cmap=plt.cm.Reds)
plt.gca().set_aspect('auto')
plt.xlabel("Words")
plt.ylabel("Documents")

inaugural_lda = LDA(data, topics=10, vocab=vocab_size)

inaugural_lda.fit()
inaugural_lda.plot_words_per_topic()

inaugural_lda.plot_word_dist()

# Above diagram is wrong for Document words per topics

plt.clf()
fig, ax = plt.subplots(1,1)
words = inaugural_lda.Z.value
topic_dist = dict()
for k_i in words:
    for k in k_i:
        if k not in topic_dist:
            topic_dist[k] = 0
        topic_dist[k] += 1
ax.bar(topic_dist.keys(), topic_dist.values())
ax.set_xlabel("Topics")
ax.set_ylabel("Counts")
ax.set_title("Document words per topics")
plt.show()

for i, t in enumerate(inaugural_lda.phi.value):
    print "Topic %i : " % i, ", ".join(idwords[w_] for w_ in np.argsort(t[0])[-10:] if w_ < vocab_size -1)



