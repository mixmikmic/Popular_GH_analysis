import collections
import commands
import json
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
from utils import get_video_number_of_frames
from utils import get_video_resolution
from skimage.transform import resize
import filmstrip
import cv2
import random

VIDEO_PATH = "/home/cabaf/AnetVideos/"
get_ipython().magic('matplotlib inline')

with open("activity_net.v1-2.json", "r") as fobj:
    data = json.load(fobj)

database = data["database"]
taxonomy = data["taxonomy"]
version = data["version"]

all_node_ids = [x["nodeId"] for x in taxonomy]
leaf_node_ids = []
for x in all_node_ids:
    is_parent = False
    for query_node in taxonomy:
        if query_node["parentId"]==x: is_parent = True
    if not is_parent: leaf_node_ids.append(x)
leaf_nodes = [x for x in taxonomy if x["nodeId"] in  leaf_node_ids]

vsize = commands.getoutput("du %s -lhs" % VIDEO_PATH).split("/")[0]
with open("../video_duration_info.json", "r") as fobj:
    tinfo = json.load(fobj)
total_duration = sum([tinfo[x] for x in tinfo])/3600.0

category_trimmed = []
for x in database:
    cc = []
    for l in database[x]["annotations"]:
        category_trimmed.append(l["label"])
category_trimmed_count = collections.Counter(category_trimmed)

print "ActivityNet %s" % version
print "Total number of videos: %d" % len(database)
print "Total number of instances: %d" % sum([category_trimmed_count[x] for x in category_trimmed_count])
print "Total number of nodes in taxonomy: %d" % len(taxonomy)
print "Total number of leaf nodes: %d" % len(leaf_nodes)
print "Total size of downloaded videos: %s" % vsize
print "Total hours of video: %0.1f" % total_duration

plt.figure(num=None, figsize=(18, 8), dpi=100)
xx = np.array(category_trimmed_count.keys())
yy = np.array([category_trimmed_count[x] for x in category_trimmed_count])
xx_idx = yy.argsort()[::-1]
plt.bar(range(len(xx)), yy[xx_idx], color=(240.0/255.0,28/255.0,1/255.0))
plt.ylabel("Number of videos per activity ")
plt.xticks(range(len(xx)), xx[xx_idx], rotation="vertical", size="small")
plt.title("ActivityNet VERSION 1.2 - Activity Detection")
plt.show()

def get_random_video_from_activity(database, activity, subset="validation"):
    videos = []
    for x in database:
        if database[x]["subset"] != subset: continue
        xx = random.choice(database[x]["annotations"])
        yy = []
        if xx["label"]==activity:
            for l in database[x]["annotations"]:
                yy.append({"videoid": x, "duration": database[x]["duration"],
                           "start_time": l["segment"][0], "end_time": l["segment"][1]})
            videos.append(yy)
    return random.choice(videos)

for ll in leaf_nodes[::-1]:
    activity = ll["nodeName"]
    keepdoing = True
    while keepdoing:
        try:
            video = get_random_video_from_activity(database, activity)
            img_montage = filmstrip.get_film_strip_from_video(video)
            assert img_montage is not None, "None returned"
            keepdoing = False
        except:
            keepdoing = True
    plt.figure(num=None, figsize=(18, 4), dpi=100)
    plt.imshow(np.uint8(img_montage)), plt.title("%s" % activity)
    plt.axis("off")
    plt.show()



