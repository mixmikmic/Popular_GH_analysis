measurements = [
    {'city': 'Dubai', 'temperature': 33.},
    {'city': 'London', 'temperature': 12.},
    {'city': 'San Fransisco', 'temperature': 18.},
]

from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer()

vec.fit_transform(measurements).toarray()



vec.get_feature_names()

from sklearn.feature_extraction.text import CountVectorizer

get_ipython().magic('pinfo CountVectorizer')

vectorizer = CountVectorizer(min_df=1)
vectorizer 

corpus = [
    'This is the first document.',
    'This is the second second document.',
    'And the third one.',
    'Is this the first document?',
]
X = vectorizer.fit_transform(corpus)
X                              

X.toarray()

analyze = vectorizer.build_analyzer()
analyze("This is a text document to analyze.")

vectorizer.get_feature_names()

vectorizer.vocabulary_.get('document')

vectorizer.transform(['Something completely new.']).toarray()

from sklearn.feature_extraction.text import TfidfTransformer
transformer = TfidfTransformer(smooth_idf=False)
transformer   

counts = [[3, 0, 1],
          [2, 0, 0],
          [3, 0, 0],
          [4, 0, 0],
          [3, 2, 0],
          [3, 0, 2]]

tfidf = transformer.fit_transform(counts)
tfidf                         

tfidf.toarray()      

from sklearn.feature_extraction.text import TfidfVectorizer

get_ipython().magic('pinfo TfidfVectorizer')



