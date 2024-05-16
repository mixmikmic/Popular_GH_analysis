import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

df = pd.read_csv("spam.csv",encoding='latin-1')

df.head()

dict = {'ham':0,'spam':1}
df['v1'] = df['v1'].map(dict)
df.head()

del df['Unnamed: 2']
del df['Unnamed: 3']
del df['Unnamed: 4']

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
c_vec = CountVectorizer(lowercase=1,min_df=.00001,stop_words='english')
c_vec.fit(df['v2'].values)

train_df = df[0:5000]
test_df = df[5000:]
test_df.index=(range(test_df.shape[0]))
Y_train = train_df['v1'].values

def prob_y(Y_train,num_class=2):
    p_y = np.zeros([num_class,])
    n_y = np.zeros([num_class,])
    d_y = Y_train.shape[0]
    for i in range(Y_train.shape[0]):
        n_y[Y_train[i]] = n_y[Y_train[i]]+1
    p_y = n_y/d_y
    return p_y

p_y = prob_y(Y_train)
p_y

def prob_xy(c_vec,train_df,Y_train,num_class=2):
    d_y = np.zeros([num_class,])+len(c_vec.vocabulary_)
    p_xy = np.zeros([num_class,len(c_vec.vocabulary_)])
    for i in np.unique(Y_train):
        temp_df = train_df[train_df['v1']==i]
        temp_x = c_vec.transform(temp_df['v2'].values)
        n_xy = np.sum(temp_x,axis=0)+1
        d_y[i] = d_y[i]+np.sum(temp_x)
        p_xy[i] = n_xy/d_y[i] 
    return p_xy

p_xy = prob_xy(c_vec,train_df,Y_train,2)
p_xy

def classify(c_vec,test_df,p_xy,p_y,num_class=2):
    pred = []
    pre_yx = []
    for doc in test_df['v2'].values:
        temp_doc = (c_vec.transform([doc])).todense()
        temp_prob = np.zeros([num_class,])
        for i in range(num_class):
            temp_prob[i] = np.prod(np.power(p_xy[i],temp_doc))*p_y[i]
        pred.append(np.argmax(temp_prob))
    return pred

pred = classify(c_vec,test_df,p_xy,p_y,num_class=2)

def accuracy(pred,Y):
    return np.sum(pred==Y)/Y.shape[0]

Y_test = test_df['v1'].values
test_accuracy = accuracy(pred,Y_test)
print('Test Data Accuaracy = '+str(test_accuracy)) 

pred_train = classify(c_vec,train_df,p_xy,p_y,num_class=2)
print('Train Data Accuracy = '+str(accuracy(pred_train,Y_train)))



