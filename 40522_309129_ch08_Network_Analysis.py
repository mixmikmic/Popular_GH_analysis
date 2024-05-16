import numpy as np
import networkx as nx

import matplotlib.pylab as plt

get_ipython().magic('matplotlib inline')
plt.style.use('seaborn-whitegrid')
plt.rc('text', usetex = True)
plt.rc('font', family = 'times')
plt.rc('xtick', labelsize = 10) 
plt.rc('ytick', labelsize = 10) 
plt.rc('font', size = 12) 
plt.rc('figure', figsize = (12, 5))

G = nx.Graph()

# Add three edges
G.add_edge('A', 'B');
G.add_edge('A', 'C');
G.add_edge('B', 'D');
G.add_edge('B', 'E');
G.add_edge('D', 'E');

# Draw the graph
nx.draw_networkx(G, node_size = 2000, font_size = 20)

plt.axis('off');
plt.savefig("files/ch08/graph_example.png", dpi = 300, bbox_inches = 'tight')

# To create a directed graph we use DiGraph:
G = nx.DiGraph()
G.add_edge('A', 'B');
G.add_edge('A', 'C');
G.add_edge('B', 'D');
G.add_edge('B', 'E');
G.add_edge('D', 'E');
nx.draw_networkx(G, node_size = 1000, font_size = 20)
plt.axis('off');

# Create a star graph:
G = nx.Graph()
G.add_edge('A', 'C');
G.add_edge('B', 'C');
G.add_edge('D', 'C');
G.add_edge('E', 'C');
G.add_edge('F', 'C');
G.add_edge('G', 'C');
G.add_edge('H', 'C');

nx.draw_networkx(G, node_size = 2000, font_size = 20)
plt.axis('off')
plt.savefig("files/ch08/star_graph.png", dpi = 300, bbox_inches = 'tight')

fb = nx.read_edgelist("files/ch08/facebook_combined.txt")

fb_n, fb_k = fb.order(), fb.size()
fb_avg_deg = fb_k / fb_n
print 'Nodes: ', fb_n
print 'Edges: ', fb_k
print 'Average degree: ', fb_avg_deg

degrees = fb.degree().values()
degree_hist = plt.hist(degrees, 100)
plt.xlabel('Degree')
plt.ylabel('Number of nodes')
plt.title('Degree distribution')
plt.savefig("files/ch08/degree_hist_plt.png", dpi = 300, bbox_inches = 'tight')

print '# connected components of Facebook network: ', nx.number_connected_components(fb)

fb_prunned = nx.read_edgelist("files/ch08/facebook_combined.txt")
fb_prunned.remove_node('0')
print 'Remaining nodes:', fb_prunned.number_of_nodes()
print 'New # connected components:', nx.number_connected_components(fb_prunned)

fb_components = nx.connected_components(fb_prunned)
print 'Sizes of the connected components', [len(c) for c in fb_components]

# Centrality measures for the star graph:
degree = nx.degree_centrality(G)
betweenness = nx.betweenness_centrality(G)
print 'Degree centrality: ', sorted(degree.items(), key = lambda x: x[1], reverse = True)
print 'Betweenness centrality: ', sorted(betweenness.items(), key = lambda x: x[1], reverse = True)

degree_cent_fb = nx.degree_centrality(fb)
# Once we are calculated degree centrality, we sort the results to see which nodes are more central.
print 'Facebook degree centrality: ', sorted(degree_cent_fb.items(), key = lambda x: x[1], reverse = True)[:10]

fig = plt.figure(figsize = (6,5))

degree_hist = plt.hist(list(degree_cent_fb.values()), 100)
plt.xlabel('Degree centrality')
plt.ylabel('Number of nodes')
plt.title('Degree centrality histogram')
plt.savefig("files/ch08/degree_centrality_hist.png", dpi = 300, bbox_inches = 'tight')

fig = plt.figure(figsize = (6,5))
degree_hist = plt.hist(list(degree_cent_fb.values()), 100)
plt.loglog(degree_hist[1][1:], degree_hist[0], 'b', marker = 'o')
plt.ylabel('Number of nodes (log)')
plt.xlabel('Degree centrality (log)')
plt.title('Sorted nodes degree (loglog)')
plt.savefig("files/ch08/degree_centrality_hist_log.png", dpi = 300, bbox_inches = 'tight')

betweenness_fb = nx.betweenness_centrality(fb)
closeness_fb = nx.closeness_centrality(fb)
eigencentrality_fb = nx.eigenvector_centrality(fb)
print 'Facebook betweenness centrality:', sorted(betweenness_fb.items(), key = lambda x: x[1], reverse = True)[:10]
print 'Facebook closeness centrality:', sorted(closeness_fb.items(), key = lambda x: x[1], reverse = True)[:10]
print 'Facebook eigenvector centrality:', sorted(eigencentrality_fb.items(), key = lambda x: x[1], reverse = True)[:10]

def trim_degree_centrality(graph, degree = 0.01):
    g = graph.copy()
    d = nx.degree_centrality(g)
    for n in g.nodes():
        if d[n] <= degree:
            g.remove_node(n)
    return g

thr = 21.0/(fb.order() - 1.0)
print 'Degree centrality threshold:', thr

fb_trimed = trim_degree_centrality (fb , degree = thr)
print 'Remaining # nodes:', len (fb_trimed)

fb_subgraph = list(nx.connected_component_subgraphs(fb_trimed))
print 'Number of found sub graphs:', np.size(fb_subgraph)
print 'Number of nodes in the first sub graph:', len(fb_subgraph[0])

betweenness = nx.betweenness_centrality(fb_subgraph[0])
print 'Trimmed Facebook betweenness centrality: ', sorted(betweenness.items(), key = lambda x: x[1], reverse = True)[:10]

current_flow = nx.current_flow_betweenness_centrality(fb_subgraph[0])
print 'Trimmed Facebook current flow betweenness centrality:', sorted(current_flow.items(), key = lambda x: x[1], reverse = True)[:10]

fig = plt.figure(figsize = (6,6))

pos = nx.random_layout(fb)
nx.draw_networkx(fb, pos, with_labels = False)
plt.axis('off') 
plt.savefig('files/ch08/facebook_Random.png', dpi = 300, bbox_inches = 'tight')

fig = plt.figure(figsize = (6,6))
nx.draw(fb)
plt.savefig('files/ch08/facebook_Default.png', dpi = 300, bbox_inches = 'tight')

pos_fb = nx.spring_layout(fb, iterations = 1000)

fig = plt.figure(figsize = (6,6))
nsize = np.array([v for v in degree_cent_fb.values()])
cte = 500
nsize = cte*(nsize  - min(nsize))/(max(nsize)-min(nsize))
nodes=nx.draw_networkx_nodes(fb, pos = pos_fb, node_size = nsize, with_labels = True)
edges=nx.draw_networkx_edges(fb, pos = pos_fb, alpha = .1, with_labels = True)
plt.axis('off') 
plt.savefig('files/ch08/facebook_Degree.png', dpi = 300, bbox_inches = 'tight')

fig = plt.figure(figsize=(6,6))

# Betweenness Centrality
nsize = np.array([v for v in betweenness_fb.values()])
nsize = cte*(nsize  - min(nsize))/(max(nsize) - min(nsize))
nodes=nx.draw_networkx_nodes(fb, pos = pos_fb, node_size = nsize)
edges=nx.draw_networkx_edges(fb, pos = pos_fb,alpha = .1)
plt.axis('off') 
plt.savefig('files/ch08/facebook_Betweenness.png', dpi = 300, bbox_inches = 'tight')

fig = plt.figure(figsize=(6,6))

# Eigenvector Centrality
nsize = np.array([v for v in eigencentrality_fb.values()])
nsize = cte*(nsize  - min(nsize))/(max(nsize) - min(nsize))
nodes = nx.draw_networkx_nodes(fb, pos = pos_fb, node_size = nsize)
edges = nx.draw_networkx_edges(fb, pos = pos_fb, alpha = .1)
plt.axis('off') 
plt.savefig('files/ch08/facebook_Eigenvector.png', dpi = 300, bbox_inches = 'tight')

fig = plt.figure(figsize=(6,6))

# Closeness Centrality
nsize = np.array([v for v in closeness_fb.values()])
nsize = cte*(nsize  - min(nsize))/(max(nsize) - min(nsize))
nodes=nx.draw_networkx_nodes(fb, pos = pos_fb, node_size = nsize)
edges=nx.draw_networkx_edges(fb, pos = pos_fb, alpha = .1)
plt.axis('off') 
plt.savefig('files/ch08/facebook_Closeness.png', dpi = 300, bbox_inches = 'tight')

fig = plt.figure(figsize = (6,6))

# Pagerank 
pr=nx.pagerank(fb, alpha = 0.85)
nsize=np.array([v for v in pr.values()])
cte = 500
nsize = cte*(nsize  - min(nsize))/(max(nsize) - min(nsize))
nodes=nx.draw_networkx_nodes(fb, pos = pos_fb, node_size = nsize)
edges=nx.draw_networkx_edges(fb, pos = pos_fb, alpha = .1)
plt.axis('off') 
plt.savefig('files/ch08/facebook_pagerank.png', dpi = 300, bbox_inches = 'tight')

fig = plt.figure(figsize=(6,6))

# Example of ego network:
G = nx.Graph()
G.add_edge('A', 'C');
G.add_edge('A', 'B');
G.add_edge('A', 'D');
G.add_edge('A', 'E');
G.add_edge('A', 'F');
G.add_edge('A', 'G');
G.add_edge('A', 'H');
G.add_edge('A', 'I');
G.add_edge('D', 'C');
G.add_edge('E', 'F');
G.add_edge('G', 'H');
G.add_edge('G', 'I');
G.add_edge('H', 'C');
G.add_edge('H', 'D');
G.add_edge('B', 'I');
c=[1, 2, 2, 2, 2, 2, 2, 2, 2]
nx.draw_networkx(G,  with_labels = False, node_color = c)
plt.axis('off') 
plt.savefig("files/ch08/ego_graph.png")

# Automatically compute ego-network
ego_107 = nx.ego_graph(fb, '107')
print '# nodes of the ego graph 107:', len(ego_107)
print '# nodes of the ego graph 107 with radius up to 2:', len(nx.ego_graph(fb, '107', radius = 2))

import os.path

ego_id = '107'
G_107 = nx.read_edgelist(os.path.join('files/ch08/facebook','{0}.edges'.format(ego_id)), nodetype = int)    
print 'Nodes of the ego graph 107: ', len(G_107)

ego_ids = (0, 107, 348, 414, 686, 698, 1684, 1912, 3437, 3980)
ego_sizes = np.zeros((10,1))

# Fill the 'ego_sizes' vector with the size (# edges) of the 10 ego-networks in egoids:
i=0
for id in ego_ids :
    G = nx.read_edgelist(os.path.join('files/ch08/facebook','{0}.edges'.format(id)), nodetype = int)    
    ego_sizes[i] = G.size()      
    print 'size of the ego-network ', id,  ego_sizes[i] 
    i +=1

[i_max, j] = (ego_sizes == ego_sizes.max()).nonzero()
ego_max = ego_ids[i_max]
print 'The most densely connected ego-network is the one of ego:', ego_max

# Load the ego network of node 1912
G = nx.read_edgelist(os.path.join('files/ch08/facebook','{0}.edges'.format(ego_max)), nodetype = int)
G_n = G.order()
G_k = G.size()
G_avg_deg = G_k / G_n
print 'Nodes: ', G_n
print 'Edges: ', G_k
print 'Average degree: ', G_avg_deg

ego_sizes = np.zeros((10,1))
i = 0
# Fill the 'egosizes' vector with the size (# nodes) of the 10 ego-networks in egoids:
for id in ego_ids :
    G = nx.read_edgelist(os.path.join('files/ch08/facebook','{0}.edges'.format(id)), nodetype = int)    
    ego_sizes[i] = G.order()      
    print 'size of the ego-network ', id,  ego_sizes[i] 
    i += 1

[i_max, j] = (ego_sizes == ego_sizes.max()).nonzero()
ego_max = ego_ids[i_max]
print 'The largest ego-network is the one of ego: ', ego_max

G = nx.read_edgelist(os.path.join('files/ch08/facebook','{0}.edges'.format(ego_max)), nodetype = int)
G_n = G.order()
G_k = G.size()
G_avg_deg = G_k / G_n
print 'Nodes: ', G_n
print 'Edges: ', G_k
print 'Average degree: ', G_avg_deg

# Add a field 'egonet' to the nodes of the whole facebook network. 
# Default value egonet=[], meaning that this node does not belong to any ego-netowrk
for i in fb.nodes() :
    fb.node[str(i)]['egonet'] = []

# Fill the 'egonet' field with one of the 10 ego values in ego_ids:
for id in ego_ids :
    G = nx.read_edgelist(os.path.join('files/ch08/facebook','{0}.edges'.format(id)), nodetype = int)
    print id
    for n in G.nodes() :
        if (fb.node[str(n)]['egonet'] == []) :
            fb.node[str(n)]['egonet'] = [id]
        else :
            fb.node[str(n)]['egonet'].append(id)

# Compute the intersections:
S = [len(x['egonet']) for x in fb.node.values()]

print '# nodes belonging to 0 ego-network: ', sum(np.equal(S,0))
print '# nodes belonging to 1 ego-network: ', sum(np.equal(S,1))
print '# nodes belonging to 2 ego-network: ', sum(np.equal(S,2))
print '# nodes belonging to 3 ego-network: ', sum(np.equal(S,3))
print '# nodes belonging to 4 ego-network: ', sum(np.equal(S,4))
print '# nodes belonging to more than 4 ego-network: ', sum(np.greater(S,4))

# Add a field 'egocolor' to the nodes of the whole facebook network. 
# Default value egocolor=0, meaning that this node does not belong to any ego-netowrk

for i in fb.nodes() :
    fb.node[str(i)]['egocolor'] = 0
    
# Fill the 'egocolor' field with a different color number for each ego-network in ego_ids:
id_color = 1
for id in ego_ids :
    G = nx.read_edgelist(os.path.join('files/ch08/facebook','{0}.edges'.format(id)), nodetype = int)
    for n in G.nodes() :
        fb.node[str(n)]['egocolor'] = id_color
    id_color += 1 

colors = [ x['egocolor'] for x in fb.node.values()]

fig = plt.figure(figsize = (6,6))

nsize = np.array([v for v in degree_cent_fb.values()])
nsize = 500*(nsize  - min(nsize))/(max(nsize) - min(nsize))

nodes=nx.draw_networkx_nodes(fb,pos = pos_fb, cmap = plt.get_cmap('Paired'), node_color = colors, 
                             node_size = nsize, with_labels = False)
edges=nx.draw_networkx_edges(fb, pos = pos_fb, alpha = .1)
plt.axis('off') 
plt.savefig('files/ch08/facebook_Colors.png', dpi = 300, bbox_inches = 'tight')

import community
partition = community.best_partition(fb)

print "# found communities:", max(partition.values())

colors2 = [partition.get(node) for node in fb.nodes()]

fig = plt.figure(figsize = (6,6))

nsize = np.array([v for v in degree_cent_fb.values()])
nsize = cte*(nsize  - min(nsize))/(max(nsize) - min(nsize))

nodes=nx.draw_networkx_nodes(fb, pos = pos_fb, cmap = plt.get_cmap('Paired'), node_color = colors2, 
                             node_size = nsize, with_labels = False)
edges=nx.draw_networkx_edges(fb, pos = pos_fb, alpha = .1)
plt.axis('off') 
plt.savefig('files/ch08/facebook_AutoPartition.png', dpi = 300, bbox_inches = 'tight')

