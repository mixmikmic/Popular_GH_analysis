import numpy as np
from ml_helpers import ml_helpers
from redis_management import RedisManagement as rmgt
from collections import OrderedDict

mat = np.load('matrix_first_vector.npy')
redis_h = rmgt('malwares')
ml_h= ml_helpers(redis_h.redis_client)
mat.shape

from sklearn.cluster import KMeans
help(KMeans)

k_means = KMeans(n_clusters=15,n_jobs=8,precompute_distances=False)

kmeans_result = k_means.fit(mat)

k_means.labels_

k_means.cluster_centers_

labels = kmeans_result.labels_.tolist()

all_malwares = ml_h.get_all_malwares
all_malwares

for index,l in enumerate(labels):
    ml_h.set_label(all_malwares[index],'KMeans','first_vector',l)

distrib = {}
for m in all_malwares:
    try:
        distrib[redis_h.client.hget(m,'KMeans_first_vector')].append((m,redis_h.client.hget(m,'label')))
    except KeyError:
        distrib[redis_h.client.hget(m,'KMeans_first_vector')] = [(m,redis_h.client.hget(m,'label'))]

distrib

redis_h.client.hget('0404b8957c27de20bebb133d3cf0a28e30700f667f7c2f3fe7fde7e726b691cd','first_vector')

np.array(list(eval(_).values()))

np.linalg.norm(_)

redis_h.client.hget('003315b0aea2fcb9f77d29223dd8947d0e6792b3a0227e054be8eb2a11f443d9','first_vector')

np.array(list(eval(_).values()))

np.linalg.norm(_)

redis_h.client.hget('01259a104a0199b794b0c61fcfc657eb766b2caeae68d5c6b164a53a97874257','first_vector')

np.array(list(eval(_).values()))

np.linalg.norm(_)

redis_h.client.hget('0cfc34fa76228b1afc7ce63e284a23ce1cd2927e6159b9dea9702ad9cb2a6300','first_vector')

np.array(list(eval(_).values()))

np.linalg.norm(_)

redis_h.client.hget('0d8c2bcb575378f6a88d17b5f6ce70e794a264cdc8556c8e812f0b5f9c709198','first_vector')

np.array(list(eval(_).values()))

np.linalg.norm(_)

import numpy as np
from ml_helpers import ml_helpers
from redis_management import RedisManagement as rmgt
from collections import OrderedDict
from collections import Counter

redis_h = rmgt('malwares')
ml_h= ml_helpers(redis_h.redis_client)
mat_second_mat = np.load('matrix_second_vector.npy')
mat_second_mat

from sklearn.cluster import KMeans

k_m = KMeans(n_clusters=20,n_jobs=8,precompute_distances=False)
k_m.fit(mat_second_mat)

k_m.labels_

labels = k_m.labels_.tolist()

all_malwares = ml_h.get_all_malwares
all_malwares

for index,l in enumerate(labels):
    ml_h.set_label(all_malwares[index],'KMeans','second_vector',l)

distrib = {}
for m in all_malwares:
    try:
        distrib[redis_h.client.hget(m,'KMeans_second_vector')].append((m,redis_h.client.hget(m,'label')))
    except KeyError:
        distrib[redis_h.client.hget(m,'KMeans_second_vector')] = [(m,redis_h.client.hget(m,'label'))]

distrib

results={}
for k,v in distrib.items():
    c = Counter()
    for malware,label in v:
        c[label] +=1
    results[k]=c
results

redis_h.client.hget('003315b0aea2fcb9f77d29223dd8947d0e6792b3a0227e054be8eb2a11f443d9','second_vector')

redis_h.client.hget('0581a38d1dc61e0da50722cb6c4253d603cc7965c87e1e42db548460d4abdcae','second_vector')

redis_h.client.hget('09c04206b57bb8582faffb37e4ebb6867a02492ffc08268bcbc717708d1a8919','second_vector')



