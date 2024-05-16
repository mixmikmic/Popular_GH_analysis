get_ipython().magic('matplotlib inline')

import os
import random
import community

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from tribe.utils import *
from tribe.stats import *
from operator import itemgetter

## Some Helper constants
FIXTURES = os.path.join(os.getcwd(), "fixtures")
# GRAPHML  = os.path.join(FIXTURES, "benjamin@bengfort.com.graphml")
GRAPHML  = os.path.join(FIXTURES, "/Users/benjamin/Desktop/20150814T212153Z.graphml")

H = nx.Graph(name="Hello World Graph")
# Also nx.DiGraph, nx.MultiGraph, etc

# Add nodes manually, label can be anything hashable
H.add_node(1, name="Ben", email="benjamin@bengfort.com")
H.add_node(2, name="Tony", email="ojedatony1616@gmail.com")

# Can also add an iterable of nodes: H.add_nodes_from
print nx.info(H)

H.add_edge(1,2, label="friends", weight=0.832)

# Can also add an iterable of edges: H.add_edges_from

print nx.info(H)
# Clearing a graph is easy
H.remove_node(1)
H.clear()

H = nx.erdos_renyi_graph(100, 0.20)

print H.nodes()[1:10]
print H.edges()[1:5]
print H.neighbors(3)

# For fast, memory safe iteration, use the `_iter` methods

edges, nodes = 0,0
for e in H.edges_iter(): edges += 1
for n in H.nodes_iter(): nodes += 1
    
print "%i edges, %i nodes" % (edges, nodes)

# Accessing the properties of a graph

print H.graph['name']
H.graph['created'] = strfnow()
print H.graph

# Accessing the properties of nodes and edges

H.node[1]['color'] = 'red'
H.node[43]['color'] = 'blue'

print H.node[43]
print H.nodes(data=True)[:3]

# The weight property is special and should be numeric
H.edge[0][34]['weight'] = 0.432
H.edge[0][36]['weight'] = 0.123

print H.edge[34][0]



# Accessing the highest degree node
center, degree = sorted(H.degree().items(), key=itemgetter(1), reverse=True)[0]

# A special type of subgraph
ego = nx.ego_graph(H, center)

pos = nx.spring_layout(H)
nx.draw(H, pos, node_color='#0080C9', edge_color='#cccccc', node_size=50)
nx.draw_networkx_nodes(H, pos, nodelist=[center], node_size=100, node_color="r")
plt.show()

# Other subgraphs can be extracted with nx.subgraph

# Finding the shortest path
H = nx.star_graph(100)
print nx.shortest_path(H, random.choice(H.nodes()), random.choice(H.nodes()))

pos = nx.spring_layout(H)
nx.draw(H, pos)
plt.show()

# Preparing for Data Science Analysis
print nx.to_numpy_matrix(H)
# print nx.to_scipy_sparse_matrix(G)

G = nx.read_graphml(GRAPHML) # opposite of nx.write_graphml

print nx.info(G)

# Generate a list of connected components
# See also nx.strongly_connected_components
for component in nx.connected_components(G):
    print len(component)

len([c for c in nx.connected_components(G)])

# Get a list of the degree frequencies
dist = FreqDist(nx.degree(G).values())
dist.plot()

# Compute Power log sequence
degree_sequence=sorted(nx.degree(G).values(),reverse=True) # degree sequence

plt.loglog(degree_sequence,'b-',marker='.')
plt.title("Degree rank plot")
plt.ylabel("degree")
plt.xlabel("rank")

# Graph Properties
print "Order: %i" % G.number_of_nodes()
print "Size: %i" % G.number_of_edges()

print "Clustering: %0.5f" % nx.average_clustering(G)

print "Transitivity: %0.5f" % nx.transitivity(G)

hairball = nx.subgraph(G, [x for x in nx.connected_components(G)][0])
print "Average shortest path: %0.4f" % nx.average_shortest_path_length(hairball)

# Node Properties
node = 'benjamin@bengfort.com' # Change to an email in your graph
print "Degree of node: %i" % nx.degree(G, node)
print "Local clustering: %0.4f" % nx.clustering(G, node)

def nbest_centrality(graph, metric, n=10, attribute="centrality", **kwargs):
    centrality = metric(graph, **kwargs)
    nx.set_node_attributes(graph, attribute, centrality)
    degrees = sorted(centrality.items(), key=itemgetter(1), reverse=True)
    
    for idx, item in enumerate(degrees[0:n]):
        item = (idx+1,) + item
        print "%i. %s: %0.4f" % item
    
    return degrees

degrees = nbest_centrality(G, nx.degree_centrality, n=15)

# centrality = nx.betweenness_centrality(G)
# normalized = nx.betweenness_centrality(G, normalized=True)
# weighted   = nx.betweenness_centrality(G, weight="weight")

degrees = nbest_centrality(G, nx.betweenness_centrality, n=15)

# centrality = nx.closeness_centrality(graph)
# normalied  = nx.closeness_centrality(graph, normalized=True)
# weighted   = nx.closeness_centrality(graph, distance="weight")

degrees = nbest_centrality(G, nx.closeness_centrality, n=15)

# centrality = nx.eigenvector_centality(graph)
# centrality = nx.eigenvector_centrality_numpy(graph)

degrees = nbest_centrality(G, nx.eigenvector_centrality_numpy, n=15)

print nx.density(G)

for subgraph in nx.connected_component_subgraphs(G):
    print nx.diameter(subgraph)
    print nx.average_shortest_path_length(subgraph)

partition = community.best_partition(G)
print "%i partitions" % len(set(partition.values()))
nx.set_node_attributes(G, 'partition', partition)

pos = nx.spring_layout(G)
plt.figure(figsize=(12,12))
plt.axis('off')

nx.draw_networkx_nodes(G, pos, node_size=200, cmap=plt.cm.RdYlBu, node_color=partition.values())
nx.draw_networkx_edges(G,pos, alpha=0.5)

nx.draw(nx.erdos_renyi_graph(20, 0.20))
plt.show()

# Generate the Graph
G=nx.davis_southern_women_graph()
# Create a Spring Layout
pos=nx.spring_layout(G)

# Find the center Node
dmin=1
ncenter=0
for n in pos:
    x,y=pos[n]
    d=(x-0.5)**2+(y-0.5)**2
    if d<dmin:
        ncenter=n
        dmin=d

# color by path length from node near center
p=nx.single_source_shortest_path_length(G,ncenter)

# Draw the graph
plt.figure(figsize=(8,8))
nx.draw_networkx_edges(G,pos,nodelist=[ncenter],alpha=0.4)
nx.draw_networkx_nodes(G,pos,nodelist=p.keys(),
                       node_size=90,
                       node_color=p.values(),
                       cmap=plt.cm.Reds_r)



