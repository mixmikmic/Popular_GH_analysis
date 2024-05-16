import http.client
import urllib.request
import urllib.parse
import urllib.error
import base64
import json
from gensim import utils

# Fill in
ACC_KEY = "????????????????????????????"


class MSCognitiveServices():
    def __init__(self, subscription_key):
        if isinstance(subscription_key, str):
            self.skey = subscription_key
            self.sample = ["I like this", "This is incredibly bad"]
        else:
            raise Exception("Key should be a string, e.g. 2ba4...")

    def send_req(self, req_list):
        if not isinstance(req_list, list):
            raise Exception("Supply a list with fewer than 100 elements")
            
        headers = {'Content-Type': 'application/json', 
                   'Ocp-Apim-Subscription-Key':'%s' % self.skey}

        try:
            documents = [{'id': no, 'text': req} for no, req in enumerate(req_list)]
            request_str = '{"documents":' + json.dumps(documents) + '}'
            conn = http.client.HTTPSConnection('westus.api.cognitive.microsoft.com')
            conn.request('POST', '/text/analytics/v2.0/sentiment', request_str, headers)
            response = conn.getresponse().read()
            data = json.loads(response.decode('utf-8'))
            out = [float(row['score'] > 0.5) for row in data['documents']]
            conn.close()
            return out
        except Exception as e:
            print(e)
        return [None] * len(req_list)

    def run_example(self):
        for x in zip(self.send_req(self.sample), self.sample):
            print(x)


msSenti = MSCognitiveServices(ACC_KEY)
msSenti.run_example()

SUBSAMPLE = 50000  # will be 100,000 total
def file_to_list(fname, lim):
    with utils.smart_open(fname) as f:
        i = 1
        for rev in f:
            yield rev
            i+=1
            if i > lim:
                return   
                       
# Data to test MS
good_reviews = []
bad_reviews = []

for f in ['test_good_reviews.txt',
          'test_bad_reviews.txt']:
    for review in file_to_list(f, SUBSAMPLE):
        if "good" in f:
            good_reviews.append(utils.to_unicode(review))
        elif "bad" in f:
            bad_reviews.append(utils.to_unicode(review))
        else:
            raise Exception

# Chunk into a list of lists for speed
counter = 0
good_scores = []
for r in (good_reviews[x:x+100] for x in range(0, len(good_reviews), 100)):
    good_scores.extend(msSenti.send_req(r))
    counter += len(r)
    #print(counter)
    
bad_scores = []
for r in (bad_reviews[x:x+100] for x in range(0, len(bad_reviews), 100)):
    bad_scores.extend(msSenti.send_req(r))   
    counter += len(r)
    #print(counter)
    
# Accuracy
acc = (good_scores.count(1.0) + bad_scores.count(0.0))/(SUBSAMPLE*2)
acc  # 0.7925

# Domain specific knowledge? Test on IMDB?

