import numpy as np
from ml_helpers import ml_helpers
from redis_management import RedisManagement as rmgt
from collections import OrderedDict

from sklearn.cluster import DBSCAN

help(DBSCAN)

mat = np.load('matrix_first_vector.npy')
redis_h = rmgt('malwares')
ml_h= ml_helpers(redis_h.redis_client)
mat.shape

dbscan = DBSCAN(eps=0.001,min_samples=1, metric="euclidean",n_jobs=8)

dbscan.fit(mat)

labels=dbscan.labels_.tolist()

all_malwares = ml_h.get_all_malwares
for index,l in enumerate(labels):
    ml_h.set_label(all_malwares[index],'DBscan','first_vector',l)

distrib = {}
for m in all_malwares:
    try:
        distrib[redis_h.client.hget(m,'DBscan_first_vector')].append((m,redis_h.client.hget(m,'label')))
    except KeyError:
        distrib[redis_h.client.hget(m,'DBscan_first_vector')] = [(m,redis_h.client.hget(m,'label'))]

distrib

distrib.keys()

[(k,len(v)) for k,v in distrib.items()]

distrib[b'4']

distrib[b'0']

distrib[b'1']

import numpy as np
from ml_helpers import ml_helpers
from redis_management import RedisManagement as rmgt
from collections import OrderedDict
from sklearn.cluster import DBSCAN

mat_second_vector = np.load('matrix_second_vector.npy')
redis_h = rmgt('malwares')
ml_h= ml_helpers(redis_h.redis_client)
mat_second_vector.shape

dbscan = DBSCAN(eps=0.01,min_samples=1, metric="euclidean",n_jobs=8)

dbscan.fit(mat_second_vector)

labels=dbscan.labels_.tolist()

all_malwares = ml_h.get_all_malwares
for index,l in enumerate(labels):
    ml_h.set_label(all_malwares[index],'DBscan','second_vector',l)

distrib = {}
for m in all_malwares:
    try:
        distrib[redis_h.client.hget(m,'DBscan_second_vector')].append((m,redis_h.client.hget(m,'label')))
    except KeyError:
        distrib[redis_h.client.hget(m,'DBscan_second_vector')] = [(m,redis_h.client.hget(m,'label'))]

distrib.keys()

sorted([(k,len(v)) for k,v in distrib.items()],key= lambda x: x[1],reverse=True )

distrib[b'1']

distrib[b'3']

distrib[b'39']

distrib[b'38']

distrib[b'105']

distrib[b'55']

distrib[b'45']

distrib[b'37']

redis_h.client.hget('e3892d2d9f87ea848477529458d025898b24a6802eb4df13e96b0314334635d0','second_vector')

redis_h.client.hget('fcfdcbdd60f105af1362cfeb3decbbbbe09d5fc82bde6ee8dfd846b2b844f972','second_vector')

distrib[b'100']

redis_h.client.hget('6c803aac51038ce308ee085f2cd82a055aaa9ba24d08a19efb2c0fcfde936c34','second_vector')

redis_h.client.hget('6217cebf11a76c888cc6ae94f54597a877462ed70da49a88589a9197173cc072','second_vector')



