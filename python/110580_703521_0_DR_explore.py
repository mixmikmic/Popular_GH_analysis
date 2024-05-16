import pandas as pd
from glob import glob
import os
import cv2
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

input_path = "../input/"

def load_df(path):    
    def get_filename(image_id):
        return os.path.join(input_path, "train", image_id + ".jpeg")

    df_node = pd.read_csv(path)
    df_node["file"] = df_node["image"].apply(get_filename)
    df_node = df_node.dropna()
    
    return df_node

df = load_df(os.path.join(input_path, "trainLabels.csv"))
len(df)

df.head()

import math

def get_filelist(level=0):
    return df[df['level'] == level]['file'].values

def subplots(filelist):
    plt.figure(figsize=(16, 12))
    ncol = 3
    nrow = math.ceil(len(filelist) // ncol)
    
    for i in range(0, len(filelist)):
        plt.subplot(nrow, ncol, i + 1)
        img = cv2.imread(filelist[i])
        plt.imshow(img)

filelist = get_filelist(level=0)
subplots(filelist[:9])

filelist = get_filelist(level=1)
subplots(filelist[:9])

filelist = get_filelist(level=2)
subplots(filelist[:9])

filelist = get_filelist(level=3)
subplots(filelist[:9])

filelist = get_filelist(level=4)
subplots(filelist[:9])

Counter(df['level'])

plt.hist(df['level'], bins=5)

