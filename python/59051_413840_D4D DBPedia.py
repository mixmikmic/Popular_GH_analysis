import requests
import json
import pandas as pd
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as colors

metrics = defaultdict(float)  # perf measurements

# use this function to run dbpedia spotlight on some text and get back dbpedia URIs
def dbpedia_spotlight(text, confidence, blacklist_types=[], sim_threshold=0):
    t0 = time.time()
    url = "http://spotlight.sztaki.hu:2222/rest/annotate"
    try:
        r = requests.get(url, params={ 
                'text': text,
                'support': 20,
                'confidence': confidence
            }, headers={'Accept': 'application/json'}, timeout=5)
    except requests.exceptions.Timeout:
        metrics['request_get'] += time.time() - t0
        print "timed out"
        return []
    
    metrics['request_get'] += time.time() - t0
    try:
        result = r.json()
    except:
        print "no response for text: %s" % text
        return []
    dbpedia_resources = []
    if 'Resources' not in result: return []
    for resource in result['Resources']:
        if float(resource['@similarityScore']) < sim_threshold: continue
        resource_types = [ r.lower() for r in resource['@types'].split(',') ]
        if len(set(blacklist_types).intersection(set(resource_types))) == 0:
            dbpedia_resources.append(resource)
    return dbpedia_resources

file_path = "/mnt/sda1/Datasets/D4D/assemble/breitbart/hist_natsec.json"  # any of the breitbart docs will work
posts = []
with open(file_path, 'r') as fin:
    posts = json.load(fin)
    
print len(posts)
posts_df = pd.DataFrame(posts)

# this dictionary is keyed by tags and has a list of post indices as values
direct_tags = defaultdict(list)
# this dictionary is keyed by post index and has a list of tags
content_resources = defaultdict(list)

# WARNING: This takes time for a full run!  Each of the 32k posts will be sent to a server for annotation.  
# If you'd like, you can run dbpedia spotlight locally (there's a docker container as well).
# The results below are from a run on the first 100 articles only
for idx, title in enumerate(posts_df['lead'][:100]):
    dbpedia_resources = dbpedia_spotlight(title, 0.35)
    for resource in dbpedia_resources:
        content_resources[idx].append(resource)
        direct_tags[resource['@URI']].append(idx)

# sort by number of articles the tag appears in
sorted_tags = sorted(direct_tags.keys(), key=lambda x: len(direct_tags[x]), reverse=True)

# check out the top 50 tags by article count
for t in sorted_tags[:50]:
    print t, len(direct_tags[t])

# build co-occurence matrix of top 50 tags
co_occuring_tags = []
idx = 0
for t1 in sorted_tags[:50]:
    row = []
    # filtering out bad tags ("The Times" was being tagged as New York Times which was rarely correct)
    if t1 in ["http://dbpedia.org/resource/The_New_York_Times", 
              "http://dbpedia.org/resource/-elect"] : continue
    for t2 in sorted_tags[:50]:
        if t2 in ["http://dbpedia.org/resource/The_New_York_Times", 
              "http://dbpedia.org/resource/-elect"] : continue
        s1 = set(direct_tags[t1])
        s2 = set(direct_tags[t2])
        if t1 == t2: 
            row.append(0.)
        else:
            # appending (2*intersection of set1 and set2) / (size of set1 + size of set2)
            row.append(float(2*len(s1.intersection(s2))) / float(len(s1)+len(s2)))
    co_occuring_tags.append(row)

np.set_printoptions(suppress=True)
plt.rcParams['figure.figsize'] = (150.0, 150.0)

get_ipython().magic('matplotlib inline')

plt.rcParams['figure.figsize'] = (50.0, 50.0)
ax = plt.imshow(np.array(co_occuring_tags), interpolation='nearest', cmap='cool', vmin=0, vmax=1).axes

m = len(co_occuring_tags)
n = m
_ = ax.set_xticks(np.linspace(0, n-1, n))
_ = ax.set_xticklabels([ x.split('/')[-1] for x in sorted_tags[:50]], fontsize=40, rotation=-90)
_ = ax.set_yticks(np.linspace(0, m-1, m))
_ = ax.set_yticklabels([ x.split('/')[-1] for x in sorted_tags[:50]], fontsize=40)

ax.grid('on')
ax.xaxis.tick_top()

def map_tags_to_field(tags_dict, data_frame, field):
    """
    Pivot function.  Returns a dict keyed by the field values
    with values equal to the tags mentioned.
    """
    pivot_dict = defaultdict(list)
    for tag, indices in tags_dict.iteritems():
        for idx in indices:
            field_value = data_frame.loc[idx][field]
            pivot_dict[field_value].append(tag)            
    return pivot_dict

def filter_dict(d, intersect):
    """
    Filter all values of the dict by intersecting with intersect.  Assumes d is a dict of lists.
    """
    for k, v in d.iteritems():
        d[k] = list(set(v).intersection(set(intersect)))
    return d

# pivot the tag data using the data frame and build a dictionary keyed by authors with lists of tags as values
author_tags = map_tags_to_field(direct_tags, posts_df, 'authors')

# filter the tag list for each author to only show tags that are in the top 100 most frequent tags
author_tags = filter_dict(author_tags, sorted_tags[:100])

author_tags

