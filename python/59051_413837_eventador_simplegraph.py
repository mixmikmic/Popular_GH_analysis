from kafka import KafkaConsumer
import uuid
import json

consumer = KafkaConsumer(bootstrap_servers='', 
                         value_deserializer=lambda s: json.loads(s, encoding='utf-8'), 
                         auto_offset_reset='smallest', 
                         group_id=uuid.uuid4()) 

consumer.subscribe(['tweets'])

limit = 500
consumer.poll(max_records=limit)
count = 0
data = []
for msg in consumer:
    data.append(msg.value)
    count += 1
    if count >= limit:
        break

len(data)

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import networkx as nx

graph = nx.DiGraph()

for tweet in data:
    if tweet.get('retweet') == 'Y':
        name = tweet.get('name')
        original_name = tweet.get('original_name')
        followers = tweet.get('followers')
        if name not in graph: graph.add_node(name, retweets = 0)
        if original_name not in graph: 
            graph.add_node(original_name, retweets = 1)
        else:
            graph.node[original_name]['retweets'] = graph.node[original_name]['retweets'] +1
        graph.add_edge(original_name, name)    

top10_retweets = sorted([(node,graph.node[node]['retweets']) for node in graph.nodes()], key = lambda x: -x[1])[0:10]
top10_retweets

pr = nx.pagerank(graph)
colors = [pr[node] for node in graph.nodes()]
top10_pr = sorted([(k,v) for k,v in pr.items()], key = lambda x: x[1])[0:10]
label_dict = dict([(k[0],k[0]) for k in top10_pr])
top10_pr

plt.figure(figsize=(11,11))
plt.axis('off')
weights = [10*(graph.node[node]['retweets'] + 1) for node in graph.nodes()]
nx.draw_networkx(graph, node_size = weights,  width = .1, linewidths = .1, with_labels=True,
                 node_color = colors, cmap = 'RdYlBu', 
                 labels = label_dict)

consumer.close()



