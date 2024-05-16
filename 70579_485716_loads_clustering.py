import os, sys
sys.path.append("tools/")
import pandas as pd
import datetime

get_ipython().magic('matplotlib inline')
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

from make_buildings_dataset import make_buildings_dataset
from describe_clusters import describe_clusters

from plot_funcs import *

path = "COMMERCIAL/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT.part1/USA_CA_Montague-Siskiyou.County.AP.725955_TMY3/RefBldgSecondarySchoolNew2004_v1.3_7.1_5B_USA_CO_BOULDER.csv"

from make_df import make_df
df = make_df(path)

days = [day.strftime("%Y-%m-%d") for day in df.index.date]

plt.rcParams["figure.figsize"] = (30,10)

plot_range(path, "2004-01-04", "2004-12-31")

#residential = make_buildings_dataset("RESIDENTIAL/")

df = pd.read_csv("all_commercial.csv").drop("Unnamed: 0", axis = 1)

from sklearn.cluster import KMeans
cluster = KMeans(n_clusters=6)

data = df[list(str(i) for i in range(0,24))]
cluster.fit(data)

def plot_building(df_row, color = "blue"):
    if color == "hairball":
        color = (0,0,0,0.01)
    plt.plot(df_row[list(str(i) for i in range(0,24))].values[0].tolist(), color = color)

plt.rcParams["figure.figsize"] = (20,12)

cmap = plt.cm.Set1(np.linspace(0, 1, 6)).tolist()
colors = ["firebrick", "darkorange", "forestgreen", "royalblue", "mediumvioletred", "gold"]
n_plot = 231

for n_cluster in range(0, len(cluster.cluster_centers_)):
    cluster_elements = (cluster.labels_ == n_cluster)

#    fig = plt.figure()
    plt.suptitle('Commercial load clustering', fontsize=24, fontweight='bold')
    plt.subplot(n_plot)
    plt.style.use('seaborn-bright')
    plt.title("Cluster n" + str(n_cluster + 1))
    
    for row in df.ix[cluster_elements].index:
        plot_building(df.ix[[row]], color = "hairball")
    
    print("CLUSTER", str(n_cluster))
    print("total elements:", str((cluster.labels_ == n_cluster).sum()))
    for unique in df.ix[cluster.labels_ == n_cluster, "building_kind"].unique():
        print("n ", unique, "=", str(len(df.ix[cluster.labels_ == n_cluster].loc[df["building_kind"] == unique])),
             "out of", len(df.loc[df["building_kind"] == unique]))
    plt.plot(cluster.cluster_centers_[n_cluster], color = colors[n_cluster], linewidth = 3)
    
    plt.xlim([0, 23])
    plt.ylim([0, 1.1])
    plt.xlabel("Hour of the day")
    plt.ylabel("Load/Peak")
    n_plot += 1
plt.savefig("clusters.png")
plt.show()

df_row = df.ix[[1]]
a = plot_building(df.ix[[1]], color = "hairball")

for center in cluster.cluster_centers_:
    plt.plot(center)
    plt.show()

import plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True) #enables plotly offline plotting
py.tools.set_credentials_file(username='gianlucahmd', api_key=open('../plotly_key.txt', "r").read())

data = []
n = 1
colors = ["green", "red", "blue", "orange", "purple"]
for center in cluster.cluster_centers_:
    data.append(go.Scatter(
            x = list(range(0,24)),
            y = center,
            name = "cluster " + str(n),
            line = dict(color = colors[n - 1])
        ))
    n += 1

layout = go.Layout(
    title = "Loads Clustering for " + str(len(cluster.cluster_centers_)) + " clusters",
    xaxis = dict(
        title = "H of day",
        range = [0,24]
    ),
    yaxis = dict(
        title = "Scaled consumption",
        range = [0,1.2]
    ))

fig = go.Figure(data = data, layout = layout)
iplot(fig)

for i in range(0, cluster.n_clusters):
    print("N elements in cluster", str(i), "=", (cluster.labels_ == i).sum())

