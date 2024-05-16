import numpy as np
np.random.seed(1000)

import warnings
warnings.filterwarnings("ignore")

import time as tm
import pandas as pd
from scipy.signal import medfilt

from keras.models import Sequential
from keras.constraints import maxnorm
from keras.layers import Dense, Dropout
from keras.utils import np_utils

from sklearn.metrics import f1_score, confusion_matrix
from sklearn.model_selection import LeaveOneGroupOut
from sklearn import preprocessing

#Cross Val of final model
from sklearn.model_selection import cross_val_score, StratifiedKFold
from keras.wrappers.scikit_learn import KerasClassifier

training_data = pd.read_csv('../training_data.csv')
blind_data = pd.read_csv('../nofacies_data.csv')

def accuracy(conf):
    total_correct = 0.
    nb_classes = conf.shape[0]
    for i in np.arange(0,nb_classes):
        total_correct += conf[i][i]
    acc = total_correct/sum(sum(conf))
    return acc

adjacent_facies = np.array([[1], [0, 2], [1], [4], [3, 5], [4, 6, 7], [5, 7], [5, 6, 8], [6, 7]])


def accuracy_adjacent(conf, adjacent_facies):
    nb_classes = conf.shape[0]
    total_correct = 0.
    for i in np.arange(0,nb_classes):
        total_correct += conf[i][i]
        for j in adjacent_facies[i]:
            total_correct += conf[i][j]
    return total_correct / sum(sum(conf))

# 1=sandstone  2=c_siltstone   3=f_siltstone 
# 4=marine_silt_shale 5=mudstone 6=wackestone 7=dolomite
# 8=packstone 9=bafflestone
facies_colors = ['#F4D03F', '#F5B041','#DC7633','#6E2C00',
       '#1B4F72','#2E86C1', '#AED6F1', '#A569BD', '#196F3D']

facies_labels = ['SS', 'CSiS', 'FSiS', 'SiSh', 'MS',
                 'WS', 'D','PS', 'BS']
#facies_color_map is a dictionary that maps facies labels
#to their respective colors
facies_color_map = {}
for ind, label in enumerate(facies_labels):
    facies_color_map[label] = facies_colors[ind]

def label_facies(row, labels):
    return labels[ row['Facies'] -1]
    
training_data.loc[:,'FaciesLabels'] = training_data.apply(lambda row: label_facies(row, facies_labels), axis=1)

X = training_data.drop(['Formation', 'Well Name', 'Facies', 'FaciesLabels'], axis=1).values
y = training_data['Facies'].values - 1
X_blind = blind_data.drop(['Formation', 'Well Name'], axis=1).values
wells = training_data["Well Name"].values

scaler = preprocessing.RobustScaler().fit(X)
X_scaled = scaler.transform(X)

def DNN():
    # Model
    model = Sequential()
    model.add(Dense(205, input_dim=8, activation='relu',W_constraint=maxnorm(5)))
    model.add(Dropout(0.1))
    model.add(Dense(69, activation='relu',W_constraint=maxnorm(5)))
    model.add(Dropout(0.1))
    model.add(Dense(69, activation='relu'))
    model.add(Dense(9, activation='softmax'))
    # Compilation
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

logo = LeaveOneGroupOut()
t0 = tm.time()
f1s_ls = []
acc_ls = []
adj_ls = []

for train, test in logo.split(X_scaled, y, groups=wells):
    well_name = wells[test[0]]
    X_tr = X_scaled[train]
    X_te = X_scaled[test]
   
    #convert y array into categories matrix
    classes = 9
    y_tr = np_utils.to_categorical(y[train], classes)
    
    # Method initialization
    NN = DNN()
    
    # Training
    NN.fit(X_tr, y_tr, nb_epoch=15, batch_size=5, verbose=0) 
    
    # Predict
    y_hat = NN.predict_classes(X_te, verbose=0)
    y_hat = medfilt(y_hat, kernel_size=7)
    
    try:
        f1s = f1_score(y[test], y_hat, average="weighted", labels=[0, 1, 2, 3, 4, 5, 6, 7, 8])
    except:
        f1s = 0

    try:
        conf = confusion_matrix(y[test], y_hat, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8])
        acc = accuracy(conf) # similar to f1 micro
    except:
        acc = 0

    try:
        acc_adj = accuracy_adjacent(conf, adjacent_facies)
    except:
        acc_adj = 0

    f1s_ls += [f1s]
    acc_ls += [acc]
    adj_ls += [acc_adj]
    print("{:>20s} f1_weigthted:{:.3f} | acc:{:.3f} | acc_adj:{:.3f}".format(well_name, f1s, acc, acc_adj))

t1 = tm.time()
print("Avg F1", np.average(f1s_ls)*100, "Avg Acc", np.average(acc_ls)*100, "Avg Adj", np.average(adj_ls)*100)
print("Blind Well Test Run Time:",'{:f}'.format((t1-t0)), "seconds")

#Another robustness test of the model using statified K fold
X_train = X_scaled
Y_train = np_utils.to_categorical(y, classes)
t2 = tm.time()
estimator = KerasClassifier(build_fn=DNN, nb_epoch=15, batch_size=5, verbose=0)
skf = StratifiedKFold(n_splits=5, shuffle=True)
results_dnn = cross_val_score(estimator, X_train, Y_train, cv= skf.get_n_splits(X_train, Y_train))
print (results_dnn)
t3 = tm.time()
print("Cross Validation Run Time:",'{:f}'.format((t3-t2)), "seconds")

NN = DNN()
NN.fit(X_train, Y_train, nb_epoch=15, batch_size=5, verbose=0)

y_predicted = NN.predict_classes(X_train, verbose=0)
y_predicted = medfilt(y_predicted, kernel_size=7)

f1s = f1_score(y, y_predicted, average="weighted")
Avgf1s = np.average(f1s_ls)*100
print ("f1 training error: ", '{:f}'.format(f1s))
print ("f1 test error: ", '{:f}'.format(Avgf1s))

x_blind = scaler.transform(X_blind)
y_blind = NN.predict_classes(x_blind, verbose=0)
y_blind = medfilt(y_blind, kernel_size=7)
blind_data["Facies"] = y_blind + 1  # return the original value (1-9)

blind_data.to_csv("J_Lowe_Submission.csv")

