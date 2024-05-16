import re
from collections import Counter
import itertools

import numpy as np
import pandas as pd

from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import matplotlib.pyplot as plt
plt.style.use('ggplot')
get_ipython().magic('matplotlib inline')

items = pd.read_csv("item_to_id.csv", index_col='Item_id')
items.sort_index(inplace=True)
items.head()

purchase_history = pd.read_csv("purchase_history.csv")
purchase_history.head()

def item_counts_by_user(same_user_df):
    # 'sum' here is adding two lists into one big list
    all_item_ids = same_user_df['id'].str.split(',').sum()
    # transform from string to int, make it easier to be sorted later
    return pd.Series(Counter(int(id) for id in all_item_ids))

user_item_counts = purchase_history.groupby("user_id").apply(item_counts_by_user).unstack(fill_value=0)

user_item_counts.shape

# each row in user_item_counts represents one user
# each column in user_item_counts represents one item
# [u,i] holds the number which user 'u' boughts item 'i'
user_item_counts.sample(5)

# we assume each "item id" in the purchase history stands for 'item_count=1'
user_item_total = user_item_counts.sum(axis=1)
print "custom who bought most in lifetime is: {}, and he/she bought {} items".format(user_item_total.argmax(),user_item_total.max())

max_user_byitem = user_item_counts.apply(lambda s: pd.Series([s.argmax(), s.max()], index=["max_user", "max_count"]))
max_user_byitem = max_user_byitem.transpose()
max_user_byitem.index.name = "Item_id"

# join with item name
max_user_byitem = max_user_byitem.join(items).loc[:, ["Item_name", "max_user", "max_count"]]
max_user_byitem

# A is |U|*|I|, and each item is normalized
A = normalize(user_item_counts.values, axis=0)
item_item_similarity = A.T.dot(A)
item_item_similarity = pd.DataFrame(item_item_similarity,
                                    index=user_item_counts.columns,
                                    columns=user_item_counts.columns)

item_item_similarity.head() # get a feeling about the data

pca = PCA()
# rotate by PCA, making it easier to be visualized later
items_rotated = pca.fit_transform(item_item_similarity)
items_rotated = pd.DataFrame(items_rotated,
                             index=user_item_counts.columns,
                             columns=["pc{}".format(index+1) for index in xrange(items.shape[0])])

# show the total variance which can be explained by first K principle components
explained_variance_by_k = pca.explained_variance_ratio_.cumsum()
plt.plot(range(1,len(explained_variance_by_k)+1),explained_variance_by_k,marker="*")

def show_clusters(items_rotated,labels):
    """
    plot and print clustering result
    """
    fig = plt.figure(figsize=(15, 15))
    colors =  itertools.cycle (["b","g","r","c","m","y","k"])

    grps = items_rotated.groupby(labels)
    for label,grp in grps:
        plt.scatter(grp.pc1,grp.pc2,c=next(colors),label = label)

        print "*********** Label [{}] ***********".format(label)
        names = items.loc[ grp.index,"Item_name"]
        for index, name in enumerate(names):
            print "\t<{}> {}".format(index+1,name)

    # annotate
    for itemid in items_rotated.index:
        x = items_rotated.loc[itemid,"pc1"]
        y = items_rotated.loc[itemid,"pc2"]
        name = items.loc[itemid,"Item_name"]
        name = re.sub('\W', ' ', name)
        plt.text(x,y,name)

    # plt.legend(loc="best")

def cluster(n_clusters,n_components=48):
    """
    n_components=K, means use first K principle components in the clustering
    n_clusters: the number of clusters we want to cluster
    """
    print "first {} PC explain {:.1f}% variances".format(n_components,
                                                         100 * sum(pca.explained_variance_ratio_[:n_components]))

    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(items_rotated.values[:, :n_components])

    # display results
    show_clusters(items_rotated, kmeans.labels_)

# choose best K (i.e., number of clusters)
inertias = []
silhouettes = []

ks = range(2,30)
for k in ks:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(items_rotated)
    
    inertias.append(kmeans.inertia_)
    silhouettes.append(silhouette_score(items_rotated, kmeans.predict(items_rotated)))

fig = plt.figure(figsize=(10,4))
fig.add_subplot(1,2,1)
plt.plot(ks,inertias,marker='x')# want to use elbow method to find best k

fig.add_subplot(1,2,2)
plt.plot(ks,silhouettes,marker='o')# the higher the better

# use all the components
cluster(n_clusters=15)



