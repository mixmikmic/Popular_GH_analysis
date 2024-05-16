import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from copy import deepcopy

data=pd.read_csv("/home/sourojit/tfmodel/Iris.csv")

data.head()

x_point=data["SepalLength"].values
y_point=data["PetalWidth"].values
points=np.array(list(zip(x_point,y_point)))
plt.scatter(x_point,y_point,c="red")
plt.show()

clusters=3

points.shape

centroid_x=[5,5,7.5]
centroid_y=[0,1,1.5]

centroid=np.array(list(zip(centroid_x,centroid_y)))

centroid

plt.scatter(x_point,y_point,c="red")
plt.scatter(centroid_x,centroid_y,marker="*",c='g')
plt.show()

centroid_old=np.zeros(centroid.shape)

cluster=np.zeros(len(points))

error=np.linalg.norm(centroid-centroid_old)

error

while error!=0:
    for i in range(len(points)):
        distance=[np.linalg.norm(points[i]-centroids) for centroids in centroid]
        c=distance.index(min(distance))
        cluster[i]=c
    centroid_old=deepcopy(centroid)    
    for i in range(len(centroid)):
        cluster_points=[]
        for j in range(len(cluster)):
            if cluster[j]==i:
                cluster_points.append(points[j])
        centroid[i]=np.mean(cluster_points,axis=0)
        centroid[i]
    error=np.linalg.norm(centroid-centroid_old)
        

error

centroid

colors=['r','g','b']
ax = plt.subplot()
for i in range(clusters):
    cluster_points=np.array([points[j] for j in range(len(points)) if cluster[j]==i])
    ax.scatter(cluster_points[:,0],cluster_points[:,1],c=colors[i])
ax.scatter(centroid[:,0],centroid[:,1],marker="*",c="black")
plt.show()

