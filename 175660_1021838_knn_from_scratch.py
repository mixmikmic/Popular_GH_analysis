# getting the required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from matplotlib import style
from collections import Counter
import random
style.use('fivethirtyeight')

# knn method (3 neighbors by default)
def knn(data,predict,k=3):
    
    # data is the training set 
    # predict is a single test data instance
    # k is number of nearest neighbors used for classification (user parameter)
    
    if len(data)>=k:
        warnings.warn('K is set to value less than total voting groups!')
    
    # distances stores distance and label of the training points from the test point
    distances = []
    for group in data:
        for features in data[group]:
            euclidean_distance = np.linalg.norm(np.array(features)-np.array(predict))
            distances.append([euclidean_distance, group])
    
    # sort distances in increasing order and take the label of the least k among them
    votes = [i[1] for i in sorted(distances)[:k]]
    
    # find the label which occurs the most and proportion of the its occurence
    vote_result = Counter(votes).most_common(1)[0][0]
    confidence = Counter(votes).most_common(1)[0][1]/k
    return vote_result , confidence   

# train set
# 3 points each in 2 classes ('k': black, 'r':red)
dataset = {'k':[[1,2],[2,3],[3,1]] , 'r':[[6,5],[7,7],[8,6]]}

# test instance
new_features = [5,7]

result = knn(dataset,new_features,3)
print(result)

# plotting the points
[[plt.scatter(ii[0],ii[1],s=100,c=i) for ii in dataset[i]] for i in dataset]
plt.scatter(new_features[0],new_features[1],s=50,c=result[0])

plt.show()

# importing the Iris Dataset
df = pd.read_csv('Iris.csv')
species = df['Species'].unique()
df.drop(['Id'],1,inplace=True)
df.head()

# converting the dataframe to a list
full_data = df.values.tolist()

# shuffling the records
random.shuffle(full_data)

# splitting the dataset into train(80%) and test sets(20%)
test_size = 0.2
train_set = {species[0]:[],species[1]:[],species[2]:[]}
test_set = {species[0]:[],species[1]:[],species[2]:[]}
train_data = full_data[:-int(test_size*len(full_data))]
test_data = full_data[-int(test_size*len(full_data)):]

for i in test_data:
    test_set[i[-1]].append(i[:-1])
    
for i in train_data:
    train_set[i[-1]].append(i[:-1])    

correct = 0
total = 0

for group in test_set:
    for data in test_set[group]:
        vote, confidence = knn(train_set,data,k=5)
        if vote==group:
            correct +=1
        else:
            print(confidence)
        total +=1

print('Accuracy:',correct/total)

