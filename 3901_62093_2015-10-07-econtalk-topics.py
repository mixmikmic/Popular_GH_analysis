get_ipython().magic('matplotlib inline')
import pyLDAvis
import json
import sys
import cPickle

from microscopes.common.rng import rng
from microscopes.lda.definition import model_definition
from microscopes.lda.model import initialize
from microscopes.lda import utils
from microscopes.lda import model, runner

from numpy import genfromtxt 
from numpy import linalg
from numpy import array

vocab = genfromtxt('./econtalk-data/vocab.txt', delimiter=",", dtype='str').tolist()
dtm = genfromtxt('./econtalk-data/dtm.csv', delimiter=",", dtype=int)
docs = utils.docs_from_document_term_matrix(dtm, vocab=vocab)
urls = [s.strip() for s in open('./econtalk-data/urls.txt').readlines()]

dtm.shape[1] == len(vocab)

dtm.shape[0] == len(urls)

def get_title(url):
    """Scrape webpage title
    """
    import lxml.html
    t = lxml.html.parse(url)
    return t.find(".//title").text.split("|")[0].strip()

N, V = len(docs), len(vocab)
defn = model_definition(N, V)
prng = rng(12345)
state = initialize(defn, docs, prng,
                        vocab_hp=1,
                        dish_hps={"alpha": .6, "gamma": 2})
r = runner.runner(defn, docs, state, )

print "randomly initialized model:"
print " number of documents", defn.n
print " vocabulary size", defn.v
print " perplexity:", state.perplexity(), "num topics:", state.ntopics()

get_ipython().run_cell_magic('time', '', 'r.run(prng, 1)')

get_ipython().run_cell_magic('time', '', 'r.run(prng, 500)')

with open('./econtalk-data/2015-10-07-state.pkl', 'w') as f:
    cPickle.dump(state, f)

get_ipython().run_cell_magic('time', '', 'r.run(prng, 500)')

with open('./econtalk-data/2015-10-07-state.pkl', 'w') as f:
    cPickle.dump(state, f)

print "after 1000 iterations:"
print " perplexity:", state.perplexity(), "num topics:", state.ntopics()

vis = pyLDAvis.prepare(**state.pyldavis_data())
pyLDAvis.display(vis)

relevance = state.term_relevance_by_topic()
for i, topic in enumerate(relevance):
    print "topic", i, ":",
    for term, _ in topic[:10]:
        print term, 
    print 

topic_distributions = state.topic_distribution_by_document()

austrian_topic = 5
foundations_episodes = sorted([(dist[austrian_topic], url) for url, dist in zip(urls, topic_distributions)], reverse=True)
for url in [url for _, url in foundations_episodes][:20]:
    print get_title(url), url

topic_a = 0
topic_b = 1
joint_episodes = [url for url, dist in zip(urls, topic_distributions) if dist[0] > 0.18 and dist[1] > 0.18]
for url in joint_episodes:
    print get_title(url), url

def find_similar(url, max_distance=0.2):
    """Find episodes most similar to input url.
    """
    index = urls.index(url)
    for td, url in zip(topic_distributions, urls):
        if linalg.norm(array(topic_distributions[index]) - array(td)) < max_distance:
            print get_title(url), url

find_similar('http://www.econtalk.org/archives/2007/04/mike_munger_on.html')

find_similar('http://www.econtalk.org/archives/2008/09/kling_on_freddi.html')

word_dists = state.word_distribution_by_topic()

def bars(x, scale_factor=10000):
    return int(x * scale_factor) * "="

def topics_related_to_word(word, n=10):
    for wd, rel in zip(word_dists, relevance):
        score = wd[word]
        rel_words = ' '.join([w for w, _ in rel][:n]) 
        if bars(score):
            print bars(score), rel_words

topics_related_to_word('munger')

topics_related_to_word('lovely')

