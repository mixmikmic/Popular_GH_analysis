import pandas as pd
import numpy as np
import requests
import json
from sklearn.metrics.pairwise import cosine_similarity

cosine_similarity(np.array([4,0,5,3,5,0,0]).reshape(1,-1),                  np.array([0,4,0,4,0,5,0]).reshape(1,-1))

cosine_similarity(np.array([4,0,5,3,5,0,0]).reshape(1,-1),                  np.array([2,0,2,0,1,0,0]).reshape(1,-1))

cosine_similarity(np.array([-.25,0,.75,-1.25,.75,0,0])                  .reshape(1,-1),                  np.array([0,-.33,0,-.33,0,.66,0])                  .reshape(1,-1))

cosine_similarity(np.array([-.25,0,.75,-1.25,.75,0,0])                  .reshape(1,-1),                  np.array([.33,0,.33,0,-.66,0,0])                  .reshape(1,-1))

user_x = [0,.33,0,-.66,0,33,0]
user_y = [0,0,0,-1,0,.5,.5]

cosine_similarity(np.array(user_x).reshape(1,-1),                  np.array(user_y).reshape(1,-1))

user_x = [0,.33,0,-.66,0,33,0]
user_z = [0,-.125,0,-.625,0,.375,.375]

cosine_similarity(np.array(user_x).reshape(1,-1),                  np.array(user_z).reshape(1,-1))

s1 = [-1.0,0.0,0.0,0.0,1.0]
s2 = [-1.66,0.0,.33,0.0,1.33]

cosine_similarity(np.array(s1).reshape(1,-1),                  np.array(s2).reshape(1,-1))

myun = 'YOUR_USERNAME'
mypw = 'YOUR_USER_TOKEN'

my_starred_repos = []
def get_starred_by_me():
    resp_list = []
    last_resp = ''
    first_url_to_get = 'https://api.github.com/user/starred'
    first_url_resp = requests.get(first_url_to_get, auth=(myun,mypw))
    last_resp = first_url_resp
    resp_list.append(json.loads(first_url_resp.text))
    
    while last_resp.links.get('next'):
        next_url_to_get = last_resp.links['next']['url']
        next_url_resp = requests.get(next_url_to_get, auth=(myun,mypw))
        last_resp = next_url_resp
        resp_list.append(json.loads(next_url_resp.text))
        
    for i in resp_list:
        for j in i:
            msr = j['html_url']
            my_starred_repos.append(msr)

get_starred_by_me()

my_starred_repos

len(my_starred_repos)

my_starred_users = []
for ln in my_starred_repos:
    right_split = ln.split('.com/')[1]
    starred_usr = right_split.split('/')[0]
    my_starred_users.append(starred_usr)

my_starred_users

len(my_starred_users)

len(set(my_starred_users))

starred_repos = {k:[] for k in set(my_starred_users)}
def get_starred_by_user(user_name):
    starred_resp_list = []
    last_resp = ''
    first_url_to_get = 'https://api.github.com/users/'+ user_name +'/starred'
    first_url_resp = requests.get(first_url_to_get, auth=(myun,mypw))
    last_resp = first_url_resp
    starred_resp_list.append(json.loads(first_url_resp.text))
    
    while last_resp.links.get('next'):
        next_url_to_get = last_resp.links['next']['url']
        next_url_resp = requests.get(next_url_to_get, auth=(myun,mypw))
        last_resp = next_url_resp
        starred_resp_list.append(json.loads(next_url_resp.text))
        
    for i in starred_resp_list:
        for j in i:
            sr = j['html_url']
            starred_repos.get(user_name).append(sr)

for usr in list(set(my_starred_users)):
    print(usr)
    try:
        get_starred_by_user(usr)
    except:
        print('failed for user', usr)

len(starred_repos)

repo_vocab = [item for sl in list(starred_repos.values()) for item in sl]

repo_set = list(set(repo_vocab))

len(repo_set)

all_usr_vector = []
for k,v in starred_repos.items():
    usr_vector = []
    for url in repo_set:
        if url in v:
            usr_vector.extend([1])
        else:
            usr_vector.extend([0])
    all_usr_vector.append(usr_vector)

len(all_usr_vector)

df = pd.DataFrame(all_usr_vector, columns=repo_set, index=starred_repos.keys())

df

len(df.columns)

my_repo_comp = []
for i in df.columns:
    if i in my_starred_repos:
        my_repo_comp.append(1)
    else:
        my_repo_comp.append(0)

mrc = pd.Series(my_repo_comp).to_frame(myun).T

mrc

mrc.columns = df.columns

fdf = pd.concat([df, mrc])

fdf

l2 = my_starred_repos

l1 = fdf.iloc[-1,:][fdf.iloc[-1,:]==1].index.values

a = set(l1)
b = set(l2)

b.difference(a)

from sklearn.metrics import jaccard_similarity_score
from scipy.stats import pearsonr

sim_score = {}
for i in range(len(fdf)):
    ss = pearsonr(fdf.iloc[-1,:], fdf.iloc[i,:])
    sim_score.update({i: ss[0]})

sf = pd.Series(sim_score).to_frame('similarity')

sf

sf.sort_values('similarity', ascending=False)

fdf.index[5]

fdf.iloc[71,:][fdf.iloc[71,:]==1]

all_recs = fdf.iloc[[31,5,71,79],:][fdf.iloc[[31,5,71,79],:]==1].fillna(0).T

all_recs[(all_recs==1).all(axis=1)]

str_recs_tmp = all_recs[all_recs[myun]==0].copy()
str_recs = str_recs_tmp.iloc[:,:-1].copy()
str_recs

str_recs[(str_recs==1).all(axis=1)]

str_recs[str_recs.sum(axis=1)>1]



