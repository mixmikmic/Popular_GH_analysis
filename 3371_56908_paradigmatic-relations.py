from py2neo import Graph
import re, string

# connect to Neo4j instance using py2neo - default running locally
graphdb = Graph('http://neo4j:neo4j@localhost:7474/db/data')

# define some parameterized Cypher queries

# For data insertion
INSERT_QUERY = '''
    FOREACH (t IN {wordPairs} | 
        MERGE (w0:Word {word: t[0]})
        MERGE (w1:Word {word: t[1]})
        CREATE (w0)-[:NEXT_WORD]->(w1)
        )
'''

# get the set of words that appear to the left of a specified word in the text corpus
LEFT1_QUERY = '''
    MATCH (s:Word {word: {word}})
    MATCH (w:Word)-[:NEXT_WORD]->(s)
    RETURN w.word as word
'''

# get the set of words that appear to the right of a specified word in the text corpus
RIGHT1_QUERY = '''
    MATCH (s:Word {word: {word}})
    MATCH (w:Word)<-[:NEXT_WORD]-(s)
    RETURN w.word as word
'''

# define a regular expression to remove punctuation
regex = re.compile('[%s]' % re.escape(string.punctuation))
exclude = set(string.punctuation)

# convert a sentence string into a list of lists of adjacent word pairs
# arrifySentence("Hi there, Bob!) = [["hi", "there"], ["there", "bob"]]
def arrifySentence(sentence):
    sentence = sentence.lower()
    sentence = sentence.strip()
    sentence = regex.sub('', sentence)
    wordArray = sentence.split()
    tupleList = []
    for i, word in enumerate(wordArray):
        if i+1 == len(wordArray):
            break
        tupleList.append([word, wordArray[i+1]])
    return tupleList

def loadFile():
    tx = graphdb.cypher.begin()
    with open('data/ceeaus.dat', encoding='ISO-8859-1') as f:
        count = 0
        for l in f:
            params = {'wordPairs': arrifySentence(l)}
            tx.append(INSERT_QUERY, params)
            tx.process()
            count += 1
            if count > 300:
                tx.commit()
                tx = graphdb.cypher.begin()
                count = 0
    f.close()
    tx.commit()

loadFile()

# return a set of all words that appear to the left of `word`
def left1(word):
    params = {
        'word': word.lower()
    }
    tx = graphdb.cypher.begin()
    tx.append(LEFT1_QUERY, params)
    results = tx.commit()
    words = []
    for result in results:
        for line in result:
            words.append(line.word)
    return set(words)

# return a set of all words that appear to the right of `word`
def right1(word):
    params = {
        'word': word.lower()
    }
    tx = graphdb.cypher.begin()
    tx.append(RIGHT1_QUERY, params)
    results = tx.commit()
    words = []
    for result in results:
        for line in result:
            words.append(line.word)
    return set(words)

# compute Jaccard coefficient
def jaccard(a,b):
    intSize = len(a.intersection(b))
    unionSize = len(a.union(b))
    return intSize / unionSize

# we define paradigmatic similarity as the average of the Jaccard coefficents of the `left1` and `right1` sets
def paradigSimilarity(w1, w2):
    return (jaccard(left1(w1), left1(w2)) + jaccard(right1(w1), right1(w2))) / 2.0



# What is the measure of paradigmatic similarity between "school" and "university" in the corpus?
paradigSimilarity("school", "university")



