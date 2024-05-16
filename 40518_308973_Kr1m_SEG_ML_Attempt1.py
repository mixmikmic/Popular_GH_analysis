import warnings
warnings.filterwarnings("ignore")
get_ipython().magic('matplotlib inline')

import sys
sys.path.append("..")

#Import standard pydata libs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

filename = '../facies_vectors.csv'
training_data = pd.read_csv(filename)
training_data['Well Name'] = training_data['Well Name'].astype('category')
training_data['Formation'] = training_data['Formation'].astype('category')                            
training_data.describe()

#Visualize the distribution of facies for each well
wells = training_data['Well Name'].unique()

fig, ax = plt.subplots(5,2, figsize=(20,20))
for i, well in enumerate(wells):
    row = i % ax.shape[0]
    column = i // ax.shape[0]
    counts = training_data[training_data['Well Name']==well].Facies.value_counts()
    data_for_well = [counts[j] if j in counts.index else 0 for j in range(1,10)]
    ax[row, column].bar(range(1,10), data_for_well, align='center')
    ax[row, column].set_title("{well}".format(well=well))
    ax[row, column].set_ylabel("Counts")
    ax[row, column].set_xticks(range(1,10))

plt.show()
    

plt.figure(figsize=(10,10))
sns.heatmap(training_data.drop(['Formation', 'Well Name'], axis=1).corr())

dfs = []
for well in training_data['Well Name'].unique():
    df = training_data[training_data['Well Name']==well].copy(deep=True)
    df.sort_values('Depth', inplace=True)
    for col in ['PE', 'GR']:
        smooth_col = 'smooth_'+col
        df[smooth_col] = pd.rolling_mean(df[col], window=25)
        df[smooth_col].fillna(method='ffill', inplace=True)
        df[smooth_col].fillna(method='bfill', inplace=True)
    dfs.append(df)
training_data = pd.concat(dfs)
pe_mean = training_data.PE.mean()
sm_pe_mean = training_data.smooth_PE.mean()
training_data['PE'] = training_data.PE.replace({np.nan:pe_mean})
training_data['smooth_PE'] = training_data['smooth_PE'].replace({np.nan:sm_pe_mean})
formation_encoder = dict(zip(training_data.Formation.unique(), range(len(training_data.Formation.unique()))))
training_data['enc_formation'] = training_data.Formation.map(formation_encoder)

training_data.describe()

#Let's build a model
from sklearn import preprocessing
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics, cross_validation 
from classification_utilities import display_cm

#We will take a look at an F1 score for each well
n_estimators=100
learning_rate=.01
random_state=0
facies_labels = ['SS', 'CSiS', 'FSiS', 'SiSh', 'MS',
                 'WS', 'D','PS', 'BS']

title_length = 20 

wells = training_data['Well Name'].unique()
for well in wells:
    blind = training_data[training_data['Well Name']==well]
    train = training_data[(training_data['Well Name']!=well)]
    
    train_X = train.drop(['Formation', 'Well Name', 'Depth', 'Facies'], axis=1)
    train_Y = train.Facies.values
    test_X = blind.drop(['Formation', 'Well Name', 'Facies', 'Depth'], axis=1)
    test_Y = blind.Facies.values
    
    clf = AdaBoostClassifier(RandomForestClassifier(), n_estimators=200, learning_rate=learning_rate, random_state=random_state, algorithm='SAMME.R')
    clf.fit(X=train_X, y=train_Y)
    pred_Y = clf.predict(test_X)
    f1 = metrics.f1_score(test_Y, pred_Y, average='micro')
    print("*"*title_length)
    print("{well}={f1:.4f}".format(well=well,f1=f1))
    print("*"*title_length)

train_X, test_X, train_Y, test_Y = cross_validation.train_test_split(training_data.drop(['Formation', 'Well Name','Facies', 'Depth'], axis=1), training_data.Facies.values, test_size=.2)

print(train_X.shape)
print(train_Y.shape)
print(test_X.shape)
print(test_Y.shape)

clf = AdaBoostClassifier(RandomForestClassifier(), n_estimators=200, learning_rate=learning_rate, random_state=0, algorithm='SAMME.R')

clf.fit(train_X, train_Y)
pred_Y = clf.predict(test_X)
cm = metrics.confusion_matrix(y_true=test_Y, y_pred=pred_Y)
display_cm(cm, facies_labels, display_metrics=True)

validation_data = pd.read_csv("../validation_data_nofacies.csv")

dfs = []
for well in validation_data['Well Name'].unique():
    df = validation_data[validation_data['Well Name']==well].copy(deep=True)
    df.sort_values('Depth', inplace=True)
    for col in ['PE', 'GR']:
        smooth_col = 'smooth_'+col
        df[smooth_col] = pd.rolling_mean(df[col], window=25)
        df[smooth_col].fillna(method='ffill', inplace=True)
        df[smooth_col].fillna(method='bfill', inplace=True)
    dfs.append(df)
validation_data = pd.concat(dfs)

validation_data['enc_formation'] = validation_data.Formation.map(formation_encoder)
validation_data.describe()

X = training_data.drop(['Formation', 'Well Name', 'Depth','Facies'], axis=1)
Y = training_data.Facies.values
test_X = validation_data.drop(['Formation', 'Well Name', 'Depth'], axis=1)

clf = AdaBoostClassifier(RandomForestClassifier(), n_estimators=200, learning_rate=learning_rate, random_state=0)
clf.fit(X,Y)
predicted_facies = clf.predict(test_X)
validation_data['Facies'] = predicted_facies

validation_data.to_csv("Kr1m_SEG_ML_Attempt1.csv")



