from py2neo import Graph
import json
import requests
import re, string
from py2neo.packages.httpstream import http
http.socket_timeout = 9999

API_KEY = "BEST_BUY_API_KEY"
# SKU = "9439005" # Kindle
# SKU = "4642026" # Bose headphones
# SKU = "6422016" # Samsung TV
# SKU = "3656051" # Samsung washing machine
# SKU = "2498029" # Dyson vacuum

REQUEST_URL = "https://api.bestbuy.com/v1/reviews(sku={sku})?apiKey={API_KEY}&show=comment,id,rating,reviewer.name,sku,submissionTime,title&pageSize=100&page={page}&sort=comment.asc&format=json"

graph = Graph()

# Build a word adjacency graph for a comment string
INSERT_QUERY = '''
WITH split(tolower({comment}), " ") AS words
WITH [w in words WHERE NOT w IN ["the","and","i", "it", "to"]] AS text
UNWIND range(0,size(text)-2) AS i
MERGE (w1:Word {name: text[i]})
ON CREATE SET w1.count = 1 ON MATCH SET w1.count = w1.count + 1
MERGE (w2:Word {name: text[i+1]})
ON CREATE SET w2.count = 1 ON MATCH SET w2.count = w2.count + 1
MERGE (w1)-[r:NEXT]->(w2)
  ON CREATE SET r.count = 1
  ON MATCH SET r.count = r.count + 1;
'''

OPINION_QUERY = '''
MATCH p=(:Word)-[r:NEXT*1..4]->(:Word) WITH p
WITH reduce(s = 0, x IN relationships(p) | s + x.count) AS total, p
WITH nodes(p) AS text, 1.0*total/size(nodes(p)) AS weight
RETURN extract(x IN text | x.name) AS phrase, weight ORDER BY weight DESC LIMIT 10
'''

# define a regular expression to remove punctuation
regex = re.compile('[%s]' % re.escape(string.punctuation))
# exclude = set(string.punctuation)

def load_graph(product_sku):
    for i in range(1,6):
        r = requests.get(REQUEST_URL.format(sku=product_sku, API_KEY=API_KEY, page=str(i)))
        data = r.json()
        for comment in data["reviews"]:
            comments = comment["comment"].split(".")
            for sentence in comments:
                sentence = sentence.strip()
                sentence = regex.sub("", sentence)
                graph.cypher.execute(INSERT_QUERY, parameters={'comment': sentence})

def summarize_opinions():
    results = graph.cypher.execute(OPINION_QUERY)
    for result in results:
        print(str(result.phrase) + " " + str(result.weight))

graph.cypher.execute("MATCH A DETACH DELETE A;")
load_graph("4642026")
summarize_opinions()

graph.cypher.execute("MATCH A DETACH DELETE A;")
load_graph("6422016")
summarize_opinions()

graph.cypher.execute("MATCH A DETACH DELETE A;")
load_graph("9439005")
summarize_opinions()

graph.cypher.execute("MATCH A DETACH DELETE A;")
load_graph("3656051")
summarize_opinions()

graph.cypher.execute("MATCH A DETACH DELETE A;")
load_graph("2498029")
summarize_opinions()





