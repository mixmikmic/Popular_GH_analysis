# Visualization 
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

#Interactive Components
from ipywidgets import interact

# Dataset Operations and Linear Algebra 
import pandas as pd
import numpy as np
import math
from scipy import stats

# Machine Learning
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# MNIST Dataset
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)

unique, counts    = np.unique(mnist.train.labels, return_counts=True)
sortedCount       = sorted(dict(zip(unique, counts)).items(), key=lambda x: x[1],reverse=True)
sortedCountLabels = [i[0] for i in sortedCount]
sortedCountFreq   = [i[1] for i in sortedCount]

# TODO: Make more efficient.
# First we will zip the training labels with the training images
dataWithLabels = zip(mnist.train.labels, mnist.train.images)

# Now let's turn this into a dictionary where subsets of the images in respect
# to digit class are stored via the corresponding key.

# Init dataDict with keys [0,9] and empty lists.
digitDict = {}
for i in range(0,10):
    digitDict[i] = []

# Assign a list of image vectors to each corresponding digit class index.
for i in dataWithLabels:
    digitDict[i[0]].append(i[1])

# Convert the lists into numpy matricies. (could be done above, but I claim ignorace)
for i in range(0,10):
    digitDict[i] = np.matrix(digitDict[i])
    print("Digit {0} matrix shape: {1}".format(i,digitDict[i].shape))


#nImgs = digitDict[9].shape[0]
#avgImg = np.dot(digitDict[9].T, np.ones((nImgs,1)))/nImgs

def pcaVectOnIMG(dataset,elmIndex):
    X_r = PCA(n_components=2).fit(dataset)
    pcaVect = X_r.transform(dataset[elmIndex])
    origin = [[14], [14]]
    plt.figure(figsize=(5,5))
    plt.imshow(dataset[elmIndex].reshape(28,28),cmap='gray')
    plt.quiver(*origin, pcaVect[:,0], pcaVect[:,1], color=['r','b','g'], scale=10)
    plt.show()
    
z =  lambda elmIndex=0,digit=1 :pcaVectOnIMG(digitDict[digit],elmIndex)

# Will error for some values of elmIndex.
interact( z, elmIndex=[0,8000],digit=[0,9])

pcaVectOnIMG(digitDict[1],1340)

pcaVectOnIMG(digitDict[1],2313)

def plotPCAVectors(data,componentIndexVec=[0,1],nComponents=120,filterDensity=50):
    n = data.shape[0]
    meanDigit = np.dot(data.T, np.ones((n,1)))/n
    
    data = data[0::filterDensity]
    X_r = PCA(n_components=nComponents).fit(data).transform(data)
    
    print("fIndex: the first principle component in the vector")
    print("sIndex: the second principle component in the vector")
    print("digit:  which digit class we are exploring.")
    
    plt.figure(figsize=(6,6))
    ax = plt.gca()
    
    origin = [[14], [14]] # origin point
    
    plt.imshow(meanDigit.reshape(28,28),cmap='gray')
    plt.quiver(*origin, X_r[:,componentIndexVec[0]], X_r[:,componentIndexVec[1]], color=['r','b','g'], scale=13)

    plt.show()
    
z = lambda fIndex=0,sIndex=1,digit=1:plotPCAVectors(digitDict[digit],[fIndex,sIndex])
    

interact(z,fIndex=[0,119],sIndex=[0,119],digit=[0,9])

def pcaR2Vects(dataset):
    return PCA(n_components=2).fit(dataset).transform(dataset)

#pcaR2Vects(digitDict[1])

def dataWithPCA_R2Vects(dataset):
    pcaVects = pcaR2Vects(dataset)
    print(dataset.shape)
    print(pcaVects.shape)
    return zip(dataset,pcaVects)

def R2Norm(vects):
    return np.linalg.norm( pcaR2Vects(ds), 2 , axis= 1)

ds = digitDict[1]
vectMag = list(zip(list(range(0,ds.shape[0],1)),R2Norm( pcaR2Vects(ds) )))

rs = sorted(vectMag, key=lambda x: x[1])
nCases = len(rs)

def tst1(elm=0):
    plt.imshow(ds[rs[elm][0]].reshape(28,28),cmap='gray')
    plt.show()
    
interact(tst1,elm=[0,20])

interact(tst1,elm=[100,200])

def meanDigitClipped(digitClass):
    n = digitClass.shape[0]
    meanDigit = np.clip( np.dot(digitClass.T, np.ones((n,1)))/n, 0.00001,1)
    return meanDigit

def meanDigitVis(digitClass=0):
    meanImg = meanDigitClipped(digitDict[digitClass])
    plt.figure(figsize=(6,6))
    plt.imshow(meanImg.reshape(28,28),cmap='gray')
    plt.show()

interact(meanDigitVis,digitClass=[0,9])

def KMeanDict(data,nClusters):
    kmLabels = KMeans(n_clusters=nClusters, random_state=None).fit(data).labels_
    classDict = {label: data[label==kmLabels] for label in np.unique(kmLabels)}    

    for i in classDict:
        classDict[i] = np.matrix(classDict[i])
        
    return classDict

def makeSubplots(nGridRow,nGridCol,figsize=(20,20)):
    sps = []
    fig = plt.figure(figsize=figsize)
    for i in range(1,(nGridRow*nGridCol)+1):
        sps.append(fig.add_subplot(nGridRow,nGridCol,i))
    return (fig,sps)

def kVis(digit=1,nClasses=4):
    figRows,figCols = (math.ceil(nClasses/4),4)
    fig,sps = makeSubplots(figRows,figCols)
    
    kDict = KMeanDict(digitDict[digit],nClasses)
    
    for i in kDict:
        md = meanDigitClipped(kDict[i])
        sps[i].imshow(md.reshape(28,28),cmap='gray')
  
        

    plt.show()
   
interact(kVis,digit=[0,9],nClasses=[2,10])



