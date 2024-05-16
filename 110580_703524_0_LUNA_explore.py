import SimpleITK
import numpy as np
import csv
from glob import glob
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

luna_path = "../inputs"
luna_subset_path = luna_path + "/subset*/"

mhd_file_list = glob(luna_subset_path + "*.mhd")

len(mhd_file_list)

import math

def plot_mhd_file(mhd_file):
    itk_img = SimpleITK.ReadImage(mhd_file) 
    img_array = SimpleITK.GetArrayFromImage(itk_img) # z,y,x ordering
    
    print("img_array.shape = ", img_array.shape)
    
    n_images = img_array.shape[0]
    ncol = 12
    nrow = math.ceil(n_images / ncol)
    
    plt.figure(figsize=(16, 16))
    for i in range(0, n_images):
        plt.subplot(nrow, ncol, i + 1)
        plt.imshow(img_array[i], cmap=plt.cm.gray)

plot_mhd_file(mhd_file_list[0])

def load_df(path):    
    def get_filename(case):
        global mhd_file_list
        for f in mhd_file_list:
            if case in f:
                return(f)

    df_node = pd.read_csv(path)
    df_node["file"] = df_node["seriesuid"].apply(get_filename)
    df_node = df_node.dropna()
    
    return df_node

df = load_df(luna_path + "/CSVFILES/annotations.csv")

print("len(df) =", len(df))

df.head()

def plot_nodule(nodule_info):
    mhd_file = nodule_info[5]
    itk_img = SimpleITK.ReadImage(mhd_file) 
    img_array = SimpleITK.GetArrayFromImage(itk_img)  # z,y,x ordering
    origin_xyz = np.array(itk_img.GetOrigin())   # x,y,z  Origin in world coordinates (mm)
    spacing_xyz = np.array(itk_img.GetSpacing()) # spacing of voxels in world coor. (mm)
    center_xyz = (nodule_info[1], nodule_info[2], nodule_info[3])
    nodule_xyz = ((center_xyz - origin_xyz) // spacing_xyz).astype(np.int16)

    import matplotlib.patches as patches
    fig, ax = plt.subplots(1)
    ax.imshow(img_array[nodule_xyz[2]], cmap=plt.cm.gray)
    ax.add_patch(
        patches.Rectangle(
            (nodule_xyz[0] - 10, nodule_xyz[1]-10),   # (x,y)
            20,          # width
            20,          # height
            linewidth=1, edgecolor='r', facecolor='none'
        )
    )

plot_nodule(df.iloc[0])
plot_nodule(df.iloc[1])
plot_nodule(df.iloc[2])
plot_nodule(df.iloc[3])

nodule_sizes = list(df['diameter_mm'])

plt.hist(nodule_sizes, bins=30)

print(SimpleITK.ReadImage(mhd_file_list[0]))

