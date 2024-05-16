from numpy.fft import rfft
from scipy import signal

import numpy as np
import matplotlib.pyplot as plt
import plotly.plotly as py


import pandas as pd
import timeit
from sqlalchemy.sql import text
from sklearn import tree
from sklearn.model_selection import LeavePGroupsOut

#from sklearn import cross_validation
#from sklearn.cross_validation import train_test_split
from sklearn import metrics

from sklearn.tree import export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import linear_model
#import sherlock.filesystem as sfs
#import sherlock.database as sdb

from sklearn import preprocessing
from sklearn.model_selection import cross_val_score

from scipy import stats

#filename = 'training_data.csv'
filename = 'facies_vectors.csv'
training_data0 = pd.read_csv(filename)


def magic(df):
    df1=df.copy()
    b, a = signal.butter(2, 0.2, btype='high', analog=False)
    feats0=['GR','ILD_log10','DeltaPHI','PHIND','PE','NM_M','RELPOS']
    #feats01=['GR','ILD_log10','DeltaPHI','PHIND']
    #feats01=['DeltaPHI']
    #feats01=['GR','DeltaPHI','PHIND']
    feats01=['GR',]
    feats02=['PHIND']
    #feats02=[]
    for ii in feats0:
        df1[ii]=df[ii]
        name1=ii + '_1'
        name2=ii + '_2'
        name3=ii + '_3'
        name4=ii + '_4'
        name5=ii + '_5'
        name6=ii + '_6'
        name7=ii + '_7'
        name8=ii + '_8'
        name9=ii + '_9'
        xx1 = list(df[ii])
        xx_mf= signal.medfilt(xx1,9)
        x_min1=np.roll(xx_mf, 1)
        x_min2=np.roll(xx_mf, -1)
        x_min3=np.roll(xx_mf, 3)
        x_min4=np.roll(xx_mf, 4)
        xx1a=xx1-np.mean(xx1)
        xx_fil = signal.filtfilt(b, a, xx1)        
        xx_grad=np.gradient(xx1a) 
        x_min5=np.roll(xx_grad, 3)
        #df1[name4]=xx_mf
        if ii in feats01: 
            df1[name1]=x_min3
            df1[name2]=xx_fil
            df1[name3]=xx_grad
            df1[name4]=xx_mf 
            df1[name5]=x_min1
            df1[name6]=x_min2
            df1[name7]=x_min4
            #df1[name8]=x_min5
            #df1[name9]=x_min2
        if ii in feats02:
            df1[name1]=x_min3
            df1[name2]=xx_fil
            df1[name3]=xx_grad
            #df1[name4]=xx_mf 
            df1[name5]=x_min1
            #df1[name6]=x_min2 
            #df1[name7]=x_min4
    return df1

        


        
        

all_wells=training_data0['Well Name'].unique()
print all_wells

# what to do with the naans
training_data1=training_data0.copy()
me_tot=training_data1['PE'].median()
print me_tot
for well in all_wells:
    df=training_data0[training_data0['Well Name'] == well] 
    print well
    print len(df)
    df0=df.dropna()
    #print len(df0)
    if len(df0) > 0:
        print "using median of local"
        me=df['PE'].median()
        df=df.fillna(value=me)
    else:
        print "using median of total"
        df=df.fillna(value=me_tot)
    training_data1[training_data0['Well Name'] == well] =df
    

print len(training_data1)
df0=training_data1.dropna()
print len(df0)

#remove outliers
df=training_data1.copy()
print len(df)
df0=df.dropna()
print len(df0)
df1 = df0.drop(['Formation', 'Well Name', 'Depth','Facies'], axis=1)
#df=pd.DataFrame(np.random.randn(20,3))
#df.iloc[3,2]=5
print len(df1)
df2=df0[(np.abs(stats.zscore(df1))<8).all(axis=1)]
print len(df2)

def run_test(remove_well, df_train):
    
    df_test=training_data2
    blind = df_test[df_test['Well Name'] == remove_well]      
    training_data = df_train[df_train['Well Name'] != remove_well]  
    
    correct_facies_labels_train = training_data['Facies'].values
    feature_vectors = training_data.drop(['Formation', 'Well Name', 'Depth','Facies'], axis=1)
    #rf = RandomForestClassifier(max_depth = 15, n_estimators=600) 
    #rf = RandomForestClassifier(max_depth = 7, n_estimators=600)  
    rf = RandomForestClassifier(max_depth = 5, n_estimators=300,min_samples_leaf=15)
    rf.fit(feature_vectors, correct_facies_labels_train)

    correct_facies_labels = blind['Facies'].values
    features_blind = blind.drop(['Formation', 'Well Name', 'Depth','Facies'], axis=1)
    scaler = preprocessing.StandardScaler().fit(feature_vectors)
    scaled_features =feature_vectors
    predicted_random_forest = rf.predict(features_blind)

    out_f1=metrics.f1_score(correct_facies_labels, predicted_random_forest,average = 'micro')
    return out_f1

    
    


training_data2=magic(training_data1)
df_train=training_data2

wells=['CHURCHMAN BIBLE','SHANKLE','NOLAN','NEWBY','Recruit F9' ,'CROSS H CATTLE','LUKE G U','SHRIMPLIN']
av_all=[]
for remove_well in wells:
    all=[]
    print("well : %s, f1 for different runs:" % (remove_well))
    for ii in range(5):
        out_f1=run_test(remove_well,df_train)   
        if remove_well is not 'Recruit F9':
            all.append(out_f1)        
    av1=np.mean(all) 
    av_all.append(av1)
    print("average f1 is %f, 2*std is %f" % (av1, 2*np.std(all)) )
print("overall average f1 is %f" % (np.mean(av_all)))

filename = 'validation_data_nofacies.csv'
test_data = pd.read_csv(filename)

test_data1=magic(test_data)

#test_well='STUART'
test_well='CRAWFORD'

blind = test_data1[test_data1['Well Name'] == test_well]      
training_data = training_data2

correct_facies_labels_train = training_data['Facies'].values
feature_vectors = training_data.drop(['Formation', 'Well Name', 'Depth','Facies'], axis=1)
rf = RandomForestClassifier(max_depth = 14, n_estimators=2500,min_samples_leaf=15) 
rf.fit(feature_vectors, correct_facies_labels_train)

features_blind = blind.drop(['Formation', 'Well Name', 'Depth'], axis=1)
predicted_random_forest = rf.predict(features_blind)


predicted_stu=predicted_random_forest
predicted_stu

predicted_craw=predicted_random_forest
predicted_craw





