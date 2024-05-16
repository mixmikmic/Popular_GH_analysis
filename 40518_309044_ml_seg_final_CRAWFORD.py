from numpy.fft import rfft
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import plotly.plotly as py
import pandas as pd
import timeit
from sqlalchemy.sql import text
from sklearn import tree
#from sklearn.model_selection import LeavePGroupsOut
from sklearn import metrics
from sklearn.tree import export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import linear_model
#import sherlock.filesystem as sfs
#import sherlock.database as sdb
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from scipy import stats
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from collections import Counter, OrderedDict
import csv


def permute_facies_nr(predicted_super, predicted0, faciesnr):
    predicted=predicted0.copy()
    N=len(predicted)
    for ii in range(N):
        if predicted_super[ii]==1:
            predicted[ii]=faciesnr  
    return predicted

def binarify(dataset0, facies_nr):
    dataset=dataset0.copy()
    mask=dataset != facies_nr
    dataset[mask]=0
    mask=dataset == facies_nr
    dataset[mask]=1    
    return dataset


def make_balanced_binary(df_in, faciesnr, factor):
    df=df_in.copy()
    y=df['Facies'].values
    y0=binarify(y, faciesnr)
    df['Facies']=y0

    df1=df[df['Facies']==1]
    X_part1=df1.drop(['Formation', 'Well Name', 'Depth','Facies'], axis=1)
    y_part1=df1['Facies'].values
    N1=len(df1)

    df2=df[df['Facies']==0]
    X_part0=df2.drop(['Formation', 'Well Name', 'Depth','Facies'], axis=1)
    y_part0=df2['Facies'].values
    N2=len(df2)
    print "ratio now:"
    print float(N2)/float(N1)
    ratio_to_keep=factor*float(N1)/float(N2)
    print "ratio after:"
    print float(N2)/(factor*float(N1))
    dum1, X_part2, dum2, y_part2 = train_test_split(X_part0, y_part0, test_size=ratio_to_keep, random_state=42)

    tmp=[X_part1, X_part2]  
    X = pd.concat(tmp, axis=0)
    y = np.concatenate((y_part1, y_part2))
    return X, y



def phaseI_model(regime_train, correctA, go_B, clf, pred_array, pred_blind, features_blind):      
    clf.fit(regime_train,correctA)     
    predicted_B = clf.predict(go_B)
    pred_array = np.vstack((predicted_B, pred_array))   
    predicted_blind1 = clf.predict(features_blind)
    pred_blind = np.vstack((predicted_blind1, pred_blind))    
    return pred_array, pred_blind

def phaseI_model_scaled(regime_train, correctA, go_B, clf, pred_array, pred_blind, features_blind):   
    regime_train=StandardScaler().fit_transform(regime_train)
    go_B=StandardScaler().fit_transform(go_B)
    features_blind=StandardScaler().fit_transform(features_blind)
    clf.fit(regime_train,correctA)     
    predicted_B = clf.predict(go_B)
    pred_array = np.vstack((predicted_B, pred_array))
    predicted_blind1 = clf.predict(features_blind)
    pred_blind = np.vstack((predicted_blind1, pred_blind))
    return pred_array, pred_blind

def create_structure_for_regimes(df):
    allfeats=['GR','ILD_log10','DeltaPHI','PHIND','PE','NM_M','RELPOS']
    data_all = []
    for feat in allfeats:
        dff=df.groupby('Well Name').describe(percentiles=[0.1, 0.25, .5, 0.75, 0.9]).reset_index().pivot(index='Well Name', values=feat, columns='level_1')
        dff = dff.drop(['count'], axis=1)
        cols=dff.columns
        cols_new=[]
        for ii in cols:
            strin=feat + "_" + str(ii)
            cols_new.append(strin)
        dff.columns=cols_new 
        dff1=dff.reset_index()
        if feat=='GR':
            data_all.append(dff1)
        else:
            data_all.append(dff1.iloc[:,1:])
    data_all = pd.concat(data_all,axis=1)
    return data_all 


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

        


        
        

#As others have done, this is Paolo Bestagini's pre-preoccessing routine 
# Feature windows concatenation function
def augment_features_window(X, N_neig):
    
    # Parameters
    N_row = X.shape[0]
    N_feat = X.shape[1]

    # Zero padding
    X = np.vstack((np.zeros((N_neig, N_feat)), X, (np.zeros((N_neig, N_feat)))))

    # Loop over windows
    X_aug = np.zeros((N_row, N_feat*(2*N_neig+1)))
    for r in np.arange(N_row)+N_neig:
        this_row = []
        for c in np.arange(-N_neig,N_neig+1):
            this_row = np.hstack((this_row, X[r+c]))
        X_aug[r-N_neig] = this_row

    return X_aug


# Feature gradient computation function
def augment_features_gradient(X, depth):
    
    # Compute features gradient
    d_diff = np.diff(depth).reshape((-1, 1))
    d_diff[d_diff==0] = 0.001
    X_diff = np.diff(X, axis=0)
    X_grad = X_diff / d_diff
        
    # Compensate for last missing value
    X_grad = np.concatenate((X_grad, np.zeros((1, X_grad.shape[1]))))
    
    return X_grad


# Feature augmentation function
def augment_features(X, well, depth, N_neig=1):
    
    # Augment features
    X_aug = np.zeros((X.shape[0], X.shape[1]*(N_neig*2+2)))
    for w in np.unique(well):
        w_idx = np.where(well == w)[0]
        X_aug_win = augment_features_window(X[w_idx, :], N_neig)
        X_aug_grad = augment_features_gradient(X[w_idx, :], depth[w_idx])
        X_aug[w_idx, :] = np.concatenate((X_aug_win, X_aug_grad), axis=1)
    
    # Find padded rows
    padded_rows = np.unique(np.where(X_aug[:, 0:7] == np.zeros((1, 7)))[0])
    
    return X_aug, padded_rows

#X_aug, padded_rows = augment_features(X, well, depth)

#filename = 'training_data.csv'
filename = 'facies_vectors.csv'
training_data0 = pd.read_csv(filename)
filename = 'validation_data_nofacies.csv'
test_data = pd.read_csv(filename)

#blindwell='CHURCHMAN BIBLE'
#blindwell='LUKE G U'
blindwell='CRAWFORD'

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

df2a=df2[df2['Well Name'] != 'Recruit F9'] 

data_all=create_structure_for_regimes(df2a)
data_test=create_structure_for_regimes(test_data)

# based on kmeans clustering
data=[]
df = training_data0[training_data0['Well Name'] == 'ALEXANDER D'] 
data.append(df)
df = training_data0[training_data0['Well Name'] == 'LUKE G U']  
data.append(df)
df = training_data0[training_data0['Well Name'] == 'CROSS H CATTLE']  
data.append(df)
Regime_1 = pd.concat(data, axis=0)
print len(Regime_1)

data=[]
df = training_data0[training_data0['Well Name'] == 'KIMZEY A']  
data.append(df)
df = training_data0[training_data0['Well Name'] == 'NOLAN']
data.append(df)
df = training_data0[training_data0['Well Name'] == 'CHURCHMAN BIBLE']  
data.append(df)
df = training_data0[training_data0['Well Name'] == 'SHANKLE'] 
data.append(df)
Regime_2 = pd.concat(data, axis=0)
print len(Regime_2)

data=[]

df = training_data0[training_data0['Well Name'] == 'SHRIMPLIN']  
data.append(df)
df = training_data0[training_data0['Well Name'] == 'NEWBY']  
data.append(df)
df = training_data0[training_data0['Well Name'] == 'Recruit F9']  
data.append(df)
Regime_3 = pd.concat(data, axis=0)
print len(Regime_3)


df0 = test_data[test_data['Well Name'] == blindwell] 
df1 = df0.drop(['Formation', 'Well Name', 'Depth'], axis=1)

#df0 = training_data0[training_data0['Well Name'] == blindwell]  
#df1 = df0.drop(['Formation', 'Well Name', 'Depth','Facies'], axis=1)

df1a=df0[(np.abs(stats.zscore(df1))<8).all(axis=1)]
blind=magic(df1a)

#features_blind = blind.drop(['Formation', 'Well Name', 'Depth','Facies'], axis=1)
features_blind = blind.drop(['Formation', 'Well Name', 'Depth'], axis=1)

#============================================================
df0=training_data0.dropna()
df1 = df0.drop(['Formation', 'Well Name', 'Depth','Facies'], axis=1)
df1a=df0[(np.abs(stats.zscore(df1))<8).all(axis=1)]
all1=magic(df1a)
#X, y = make_balanced_binary(all1, 9,6)
for kk in range(3,4):
    X, y = make_balanced_binary(all1, 9,kk)
#============================================================
    correct_train=y

    #clf = RandomForestClassifier(max_depth = 6, n_estimators=1600)
    clf = RandomForestClassifier(max_depth = 6, n_estimators=800)
    clf.fit(X,correct_train)

    predicted_blind1 = clf.predict(features_blind)

    predicted_regime9=predicted_blind1.copy()
    print("kk is %d, nr of predictions for this regime is %d" % (kk, sum(predicted_regime9)))
    print "----------------------------------"


#features_blind = blind.drop(['Formation', 'Well Name', 'Depth'], axis=1)

#============================================================
df0=training_data0.dropna()
df1 = df0.drop(['Formation', 'Well Name', 'Depth','Facies'], axis=1)
df1a=df0[(np.abs(stats.zscore(df1))<8).all(axis=1)]
all1=magic(df1a)

for kk in range(4,6):
#for kk in range(1,6): 
    X, y = make_balanced_binary(all1, 1,kk)
    #============================================================

    #=============================================
    go_A=StandardScaler().fit_transform(X)
    go_blind=StandardScaler().fit_transform(features_blind)
    correct_train_A=binarify(y, 1)
                                        

    clf = linear_model.LogisticRegression()
    clf.fit(go_A,correct_train_A)
    predicted_blind1 = clf.predict(go_blind)

    clf = KNeighborsClassifier(n_neighbors=5)
    clf.fit(go_A,correct_train_A)                                                  
    predicted_blind2 = clf.predict(go_blind)

    clf = svm.SVC(decision_function_shape='ovo')
    clf.fit(go_A,correct_train_A)   
    predicted_blind3 = clf.predict(go_blind)

    clf = svm.LinearSVC()
    clf.fit(go_A,correct_train_A)   
    predicted_blind4 = clf.predict(go_blind)



    #####################################
    predicted_blind=predicted_blind1+predicted_blind2+predicted_blind3+predicted_blind4
    for ii in range(len(predicted_blind)):
        if predicted_blind[ii] > 3:
            predicted_blind[ii]=1
        else:
            predicted_blind[ii]=0 
        
    for ii in range(len(predicted_blind)):
        if predicted_blind[ii] == 1 and predicted_blind[ii-1] == 0 and predicted_blind[ii+1] == 0:
            predicted_blind[ii]=0
        if predicted_blind[ii] == 1 and predicted_blind[ii-1] == 0 and predicted_blind[ii+2] == 0:
            predicted_blind[ii]=0        
        if predicted_blind[ii] == 1 and predicted_blind[ii-2] == 0 and predicted_blind[ii+1] == 0:
            predicted_blind[ii]=0     
    #####################################    

    print "-------"
    predicted_regime1=predicted_blind.copy()

    #print("%c is my %s letter and my number %d number is %.5f" % ('X', 'favorite', 1, .14))
 
    print("kk is %d, nr of predictions for this regime is %d" % (kk, sum(predicted_regime1)))
    print "----------------------------------"

#features_blind = blind.drop(['Formation', 'Well Name', 'Depth'], axis=1)

#============================================================
df0=training_data0.dropna()
df1 = df0.drop(['Formation', 'Well Name', 'Depth','Facies'], axis=1)
df1a=df0[(np.abs(stats.zscore(df1))<8).all(axis=1)]
all1=magic(df1a)
for kk in range(1,6):
#for kk in range(2,4): 
    X, y = make_balanced_binary(all1, 5,kk)
    #X, y = make_balanced_binary(all1, 5,13)
    #============================================================

    go_A=StandardScaler().fit_transform(X)
    go_blind=StandardScaler().fit_transform(features_blind)
    correct_train_A=binarify(y, 1)
    #=============================================                                        

    clf = KNeighborsClassifier(n_neighbors=4,algorithm='brute')
    clf.fit(go_A,correct_train_A)
    predicted_blind1 = clf.predict(go_blind)

    clf = KNeighborsClassifier(n_neighbors=5,leaf_size=10)
    clf.fit(go_A,correct_train_A)                                                  
    predicted_blind2 = clf.predict(go_blind)

    clf = KNeighborsClassifier(n_neighbors=5)
    clf.fit(go_A,correct_train_A)   
    predicted_blind3 = clf.predict(go_blind)

    clf = tree.DecisionTreeClassifier()
    clf.fit(go_A,correct_train_A)   
    predicted_blind4 = clf.predict(go_blind)

    clf = tree.DecisionTreeClassifier()
    clf.fit(go_A,correct_train_A)   
    predicted_blind5 = clf.predict(go_blind)

    clf = tree.DecisionTreeClassifier()
    clf.fit(go_A,correct_train_A)    
    predicted_blind6 = clf.predict(go_blind)


    #####################################
    predicted_blind=predicted_blind1+predicted_blind2+predicted_blind3+predicted_blind4+predicted_blind5+predicted_blind6
    for ii in range(len(predicted_blind)):
        if predicted_blind[ii] > 4:
            predicted_blind[ii]=1
        else:
            predicted_blind[ii]=0 

    print "-------"
    predicted_regime5=predicted_blind.copy()
    print("kk is %d, nr of predictions for this regime is %d" % (kk, sum(predicted_regime5)))
    print "----------------------------------"

#features_blind = blind.drop(['Formation', 'Well Name', 'Depth'], axis=1)

#============================================================
df0=training_data0.dropna()
df1 = df0.drop(['Formation', 'Well Name', 'Depth','Facies'], axis=1)
df1a=df0[(np.abs(stats.zscore(df1))<8).all(axis=1)]
all1=magic(df1a)
for kk in range(2,5):
    X, y = make_balanced_binary(all1, 7,kk)
    #X, y = make_balanced_binary(all1, 7,13)
    #============================================================

    go_A=StandardScaler().fit_transform(X)
    go_blind=StandardScaler().fit_transform(features_blind)
    correct_train_A=binarify(y, 1)
    #=============================================                                        

    clf = KNeighborsClassifier(n_neighbors=4,algorithm='brute')
    clf.fit(go_A,correct_train_A)
    predicted_blind1 = clf.predict(go_blind)


    clf = KNeighborsClassifier(n_neighbors=5,leaf_size=10)
    clf.fit(go_A,correct_train_A)                                                  
    predicted_blind2 = clf.predict(go_blind)

    clf = KNeighborsClassifier(n_neighbors=5)
    clf.fit(go_A,correct_train_A)   
    predicted_blind3 = clf.predict(go_blind)

    clf = tree.DecisionTreeClassifier()
    clf.fit(go_A,correct_train_A)   
    predicted_blind4 = clf.predict(go_blind)

    clf = tree.DecisionTreeClassifier()
    clf.fit(go_A,correct_train_A)   
    predicted_blind5 = clf.predict(go_blind)

    clf = tree.DecisionTreeClassifier()
    clf.fit(go_A,correct_train_A)    
    predicted_blind6 = clf.predict(go_blind)


    #####################################
    predicted_blind=predicted_blind1+predicted_blind2+predicted_blind3+predicted_blind4+predicted_blind5+predicted_blind6
    for ii in range(len(predicted_blind)):
        if predicted_blind[ii] > 5:
            predicted_blind[ii]=1
        else:
            predicted_blind[ii]=0 


    #####################################    
    print "-------"
    predicted_regime7=predicted_blind.copy()
    print("kk is %d, nr of predictions for this regime is %d" % (kk, sum(predicted_regime7)))
    print "----------------------------------"

def prepare_data(Regime_1, Regime_2, Regime_3, test_data, w1, w2,w3):
    df0=Regime_1.dropna()
    df1 = df0.drop(['Formation', 'Well Name', 'Depth','Facies'], axis=1)
    df1a=df0[(np.abs(stats.zscore(df1))<8).all(axis=1)]
    df2a=magic(df1a)
    feature_names0 = ['GR', 'ILD_log10','DeltaPHI', 'PHIND', 'PE', 'NM_M', 'RELPOS', 'PHIND_1', 'PHIND_2']
    X0 = df2a[feature_names0].values
    df2a=(df1a)
    y=df2a['Facies'].values
    feature_names = ['GR', 'ILD_log10', 'DeltaPHI', 'PHIND', 'PE', 'NM_M', 'RELPOS']
    X1 = df2a[feature_names].values
    well = df2a['Well Name'].values
    depth = df2a['Depth'].values
    X2, padded_rows = augment_features(X1, well, depth)
    Xtot_train=np.column_stack((X0,X2))
    regime1A_train, regime1B_train, regime1A_test, regime1B_test = train_test_split(Xtot_train, y, test_size=w1, random_state=42)

    df0=Regime_2.dropna()
    df1 = df0.drop(['Formation', 'Well Name', 'Depth','Facies'], axis=1)
    df1a=df0[(np.abs(stats.zscore(df1))<8).all(axis=1)]
    df2a=magic(df1a)
    feature_names0 = ['GR', 'ILD_log10','DeltaPHI', 'PHIND', 'PE', 'NM_M', 'RELPOS', 'PHIND_1', 'PHIND_2']
    X0 = df2a[feature_names0].values
    df2a=(df1a)
    y=df2a['Facies'].values
    feature_names = ['GR', 'ILD_log10', 'DeltaPHI', 'PHIND', 'PE', 'NM_M', 'RELPOS']
    X1 = df2a[feature_names].values
    well = df2a['Well Name'].values
    depth = df2a['Depth'].values
    X2, padded_rows = augment_features(X1, well, depth)
    Xtot_train=np.column_stack((X0,X2))
    regime2A_train, regime2B_train, regime2A_test, regime2B_test = train_test_split(Xtot_train, y, test_size=w2, random_state=42)


    df0=Regime_3.dropna()
    df1 = df0.drop(['Formation', 'Well Name', 'Depth','Facies'], axis=1)
    df1a=df0[(np.abs(stats.zscore(df1))<8).all(axis=1)]
    df2a=magic(df1a)
    feature_names0 = ['GR', 'ILD_log10','DeltaPHI', 'PHIND', 'PE', 'NM_M', 'RELPOS', 'PHIND_1', 'PHIND_2']
    X0 = df2a[feature_names0].values
    df2a=(df1a)
    y=df2a['Facies'].values
    feature_names = ['GR', 'ILD_log10', 'DeltaPHI', 'PHIND', 'PE', 'NM_M', 'RELPOS']
    X1 = df2a[feature_names].values
    well = df2a['Well Name'].values
    depth = df2a['Depth'].values
    X2, padded_rows = augment_features(X1, well, depth)
    Xtot_train=np.column_stack((X0,X2))
    regime3A_train, regime3B_train, regime3A_test, regime3B_test = train_test_split(Xtot_train, y, test_size=w3, random_state=42)


    #df0 = training_data0[training_data0['Well Name'] == blindwell]
    #df1 = df0.drop(['Formation', 'Well Name', 'Depth','Facies'], axis=1)
    df0 = test_data[test_data['Well Name'] == blindwell] 
    df1 = df0.drop(['Formation', 'Well Name', 'Depth'], axis=1)
    df1a=df0[(np.abs(stats.zscore(df1))<8).all(axis=1)]
    df2a=magic(df1a)
    #df2a=df1a
    X0blind = df2a[feature_names0].values

    blind=df1a
    #correct_facies_labels = blind['Facies'].values
    feature_names = ['GR', 'ILD_log10', 'DeltaPHI', 'PHIND', 'PE', 'NM_M', 'RELPOS']
    X1 = blind[feature_names].values
    well = blind['Well Name'].values
    depth = blind['Depth'].values
    X2blind,  padded_rows = augment_features(X1, well, depth)

    features_blind=np.column_stack((X0blind,X2blind))
#=======================================================
    main_regime=regime2A_train
    other1=regime1A_train
    other2=regime3A_train

    main_test=regime2A_test
    other1_test=regime1A_test
    other2_test=regime3A_test

    go_B=np.concatenate((regime1B_train, regime2B_train, regime3B_train))
    correctB=np.concatenate((regime1B_test, regime2B_test, regime3B_test))
#     #===================================================
    train1= np.concatenate((main_regime, other1, other2))
    correctA1=np.concatenate((main_test, other1_test, other2_test))
#     #=================================================== 
#     train2= np.concatenate((main_regime, other2))
#     correctA2=np.concatenate((main_test, other2_test))
#     #===================================================

    #===================================================
    #train1=main_regime
    #correctA1=main_test
    train2=other1
    correctA2=other1_test   
    train3=other2
    correctA3=other2_test   

    return train1, train2, train3, correctA1, correctA2, correctA3, correctB, go_B, features_blind

def run_phaseI(train1,train2,train3,correctA1,correctA2,correctA3,correctB, go_B, features_blind):    
    pred_array=0*correctB
    pred_blind=np.zeros(len(features_blind))

    print "rf1"
    clf = RandomForestClassifier(max_depth = 5, n_estimators=2600, random_state=1)
    pred_array, pred_blind=phaseI_model(train1, correctA1, go_B, clf, pred_array, pred_blind, features_blind)
    clf = RandomForestClassifier(max_depth = 15, n_estimators=3000)
    pred_array, pred_blind=phaseI_model(train1, correctA1, go_B, clf, pred_array, pred_blind, features_blind)    
#     pred_array, pred_blind=phaseI_model(train2, correctA2, go_B, clf, pred_array, pred_blind, features_blind)
#     pred_array, pred_blind=phaseI_model(train3, correctA3, go_B, clf, pred_array, pred_blind, features_blind)
    clf = RandomForestClassifier(n_estimators=1200, max_depth = 15, criterion='entropy',
                                 max_features=10, min_samples_split=25, min_samples_leaf=5,
                                 class_weight='balanced', random_state=1)
    pred_array, pred_blind=phaseI_model(train1, correctA1, go_B, clf, pred_array, pred_blind, features_blind)
    #pred_array, pred_blind=phaseI_model(train2, correctA2, go_B, clf, pred_array, pred_blind, features_blind)
    #pred_array, pred_blind=phaseI_model(train3, correctA3, go_B, clf, pred_array, pred_blind, features_blind)
    return pred_array, pred_blind

w1=0.05
w2=0.05
w3=0.05
print "preparing data:"
#train1, train2, train3, correctA1, correctA2, correctA3, correctB, go_B, features_blind=prepare_data(Regime_1, Regime_2, Regime_3, training_data0, w1, w2,w3)
train1, train2, train3, correctA1, correctA2, correctA3, correctB, go_B, features_blind=prepare_data(Regime_1, Regime_2, Regime_3, test_data, w1, w2,w3)
print(len(correctB))
print "running phase I:"
pred_array, pred_blind = run_phaseI(train1,train2,train3,correctA1,correctA2, correctA3, correctB, go_B, features_blind)
print "prediction phase II:"
clf = RandomForestClassifier(max_depth = 8, n_estimators=3000, max_features=10, criterion='entropy',class_weight='balanced')
#clf = RandomForestClassifier(max_depth = 5, n_estimators=300, max_features=10, criterion='entropy',class_weight='balanced')
#clf = RandomForestClassifier(n_estimators=1200, max_depth = 15, criterion='entropy',
#                             max_features=10, min_samples_split=25, min_samples_leaf=5,
#                             class_weight='balanced', random_state=1)
#clf = RandomForestClassifier(n_estimators=1200, max_depth = 5, criterion='entropy',
#                             max_features=10, min_samples_split=25, min_samples_leaf=5,
#                             class_weight='balanced', random_state=1)
clf.fit(go_B,correctB)
predicted_blind_PHASE_I = clf.predict(features_blind)

print "prediction phase II-stacked:"
pa=pred_array[:len(pred_array)-1]
go_B_PHASE_II=np.concatenate((pa, go_B.transpose())).transpose()
pa1=np.median(pa,axis=0)
go_B_PHASE_II=np.column_stack((go_B_PHASE_II,pa1))
print go_B_PHASE_II.shape
feat=pred_blind[:len(pred_blind)-1]
features_blind_PHASE_II=np.concatenate((feat, features_blind.transpose())).transpose()
feat1=np.median(feat,axis=0)
features_blind_PHASE_II=np.column_stack((features_blind_PHASE_II,feat1))

#second pred
clf.fit(go_B_PHASE_II,correctB)
predicted_blind_PHASE_II = clf.predict(features_blind_PHASE_II)

#print "finished"
#out_f1=metrics.f1_score(correct_facies_labels, predicted_blind_PHASE_I, average = 'micro')
#print " f1 score on the prediction of blind:"
#print out_f1
#out_f1=metrics.f1_score(correct_facies_labels, predicted_blind_PHASE_II, average = 'micro')
#print " f1 score on the prediction of blind:"
#print out_f1
#print "finished"
#print "-----------------------------"   

print(sum(predicted_regime5))
predicted_blind_PHASE_IIa=permute_facies_nr(predicted_regime5, predicted_blind_PHASE_II, 5)
print(sum(predicted_regime7))
predicted_blind_PHASE_IIb=permute_facies_nr(predicted_regime7, predicted_blind_PHASE_IIa, 7)
print(sum(predicted_regime1))
predicted_blind_PHASE_IIc=permute_facies_nr(predicted_regime1, predicted_blind_PHASE_IIb, 1)
print(sum(predicted_regime9))
predicted_blind_PHASE_III=permute_facies_nr(predicted_regime9, predicted_blind_PHASE_IIc, 9)


print "values changed:"

print len(predicted_blind_PHASE_II)-np.count_nonzero(predicted_blind_PHASE_III==predicted_blind_PHASE_II)

predicted_blind_CRAWFORD=predicted_blind_PHASE_III
predicted_blind_CRAWFORD

x=Counter(predicted_blind_PHASE_I)
y = OrderedDict(x)
y

x=Counter(predicted_blind_PHASE_II)
y = OrderedDict(x)
y

x=Counter(predicted_blind_PHASE_III)
y = OrderedDict(x)
y



