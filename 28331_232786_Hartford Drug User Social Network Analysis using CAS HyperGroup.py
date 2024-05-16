import swat
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx

# Also import networkx used for rendering a network
import networkx as nx

get_ipython().magic('matplotlib inline')

s = swat.CAS('http://viya.mycompany.com:8777') # REST API

s.loadactionset('hypergroup')

drug_network = pd.read_csv('drug_network.csv')

drug_network['SOURCE'] = drug_network['FROM'].astype(str)
drug_network['TARGET'] = drug_network['TO'].astype(str)
drug_network.head()

if s.tableexists('drug_network').exists:
    s.CASTable('drug_network').droptable()
    
dataset = s.upload_frame(drug_network, 
                         importoptions=dict(vars=[dict(type='double'),
                                                  dict(type='double'),
                                                  dict(type='varchar'),
                                                  dict(type='varchar')]),
                          casout=dict(name='drug_network', promote=True))

dataset.columninfo()

dataset.head()

dataset.summary()

def renderNetworkGraph(filterCommunity=-1, size=18, sizeVar='_HypGrp_',
                       colorVar='', sizeMultipler=500, nodes_table='nodes',
                       edges_table='edges'):
    ''' Build an array of node positions and related colors based on community '''
    nodes = s.CASTable(nodes_table)
    if filterCommunity >= 0:
        nodes = nodes.query('_Community_ EQ %F' % filterCommunity)
    nodes = nodes.to_frame()

    nodePos = {}
    nodeColor = {}
    nodeSize = {}
    communities = []
    i = 0
    for nodeId in nodes._Value_:    
        nodePos[nodeId] = (nodes._AllXCoord_[i], nodes._AllYCoord_[i])
        if colorVar: 
            nodeColor[nodeId] = nodes[colorVar][i]
            if nodes[colorVar][i] not in communities:
                communities.append(nodes[colorVar][i])
        nodeSize[nodeId] = max(nodes[sizeVar][i],0.1)*sizeMultipler
        i += 1
    communities.sort()
  
    # Build a list of source-target tuples
    edges = s.CASTable(edges_table)
    if filterCommunity >= 0:
        edges = edges.query('_SCommunity_ EQ %F AND _TCommunity_ EQ %F' % 
                            (filterCommunity, filterCommunity))
    edges = edges.to_frame()

    edgeTuples = []
    i = 0
    for p in edges._Source_:
        edgeTuples.append( (edges._Source_[i], edges._Target_[i]) )
        i += 1
    
    # Add nodes and edges to the graph
    plt.figure(figsize=(size,size))
    graph = nx.DiGraph()
    graph.add_edges_from(edgeTuples)

    # Size mapping
    getNodeSize=[nodeSize[v] for v in graph]
    
    # Color mapping
    jet = cm = plt.get_cmap('jet')
    getNodeColor=None
    if colorVar: 
        getNodeColor=[nodeColor[v] for v in graph]
        cNorm  = colors.Normalize(vmin=min(communities), vmax=max(communities))
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
    
        # Using a figure here to work-around the fact that networkx doesn't produce a labelled legend
        f = plt.figure(1)
        ax = f.add_subplot(1,1,1)
        for community in communities:
            ax.plot([0],[0], color=scalarMap.to_rgba(community), 
                    label='Community %s' % '{:2.0f}'.format(community),linewidth=10)
        
    # Render the graph
    nx.draw_networkx_nodes(graph, nodePos, node_size=getNodeSize, 
                           node_color=getNodeColor, cmap=jet)
    nx.draw_networkx_edges(graph, nodePos, width=1, alpha=0.5)
    nx.draw_networkx_labels(graph, nodePos, font_size=11, font_family='sans-serif')
        
    if len(communities) > 0:
        plt.legend(loc='upper left',prop={'size':11})
        
    plt.title('Hartford Drug User Social Network', fontsize=30)
    plt.axis('off')
    plt.show()

# Create output table objects
edges = s.CASTable('edges', replace=True)
nodes = s.CASTable('nodes', replace=True)

dataset[['SOURCE', 'TARGET']].hypergroup(
    createOut = 'never',
    allGraphs = True,
    edges     = edges,
    vertices  = nodes
)

renderNetworkGraph()

dataset[['SOURCE', 'TARGET']].hypergroup(
    createOut = 'never',
    allGraphs = True,
    community = True,
    edges     = edges,
    vertices  = nodes
)

nodes.distinct()

nodes.summary()

topKOut = s.CASTable('topKOut', replace=True)

nodes[['_Community_']].topk(
    aggregator = 'N',
    topK       = 4,
    casOut     = topKOut
)

topKOut = topKOut.sort_values('_Rank_').head(10)
topKOut.columns

nCommunities = len(topKOut)

ind = np.arange(nCommunities)    # the x locations for the groups

plt.figure(figsize=(8, 4))
p1 = plt.bar(ind + 0.2, topKOut._Score_, 0.5, color='orange', alpha=0.75)

plt.ylabel('Vertices', fontsize=12)
plt.xlabel('Community', fontsize=12)
plt.title('Number of Nodes for the Top %s Communities' % nCommunities)
plt.xticks(ind + 0.2, topKOut._Fmtvar_)

plt.show()

nodes.query('_Community_ EQ 4').head()

edges.head()

renderNetworkGraph(colorVar='_Community_')

dataset[['SOURCE', 'TARGET']].hypergroup(
    createOut = 'never',
    community = True,
    nCommunities = 5,
    allGraphs = True,
    edges     = edges,
    vertices  = nodes
)

renderNetworkGraph(colorVar='_Community_')

dataset[['SOURCE', 'TARGET']].hypergroup(
    createOut = 'never',
    community = True,
    nCommunities = 5,
    centrality = True,
    mergeCommSmallest = True,
    allGraphs = True,
    graphPartition = True,
    scaleCentralities = 'central1', # returns centrality values closer to 1 in the center
    edges     = edges,
    vertices  = nodes
)

nodes.head()

renderNetworkGraph(colorVar='_Community_', sizeVar='_Betweenness_')

renderNetworkGraph(2, size=10, sizeVar='_CentroidAngle_', sizeMultipler=5)

s.close()

