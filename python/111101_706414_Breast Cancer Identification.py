get_ipython().magic('matplotlib inline')
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import mode,skew,skewtest

from sklearn.cross_validation import train_test_split
from sklearn import ensemble
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score

data=pd.read_csv('/Users/adityavyas/Desktop/Machine Learning and Big Data/Datasets/Breast Cancer/data.csv')

data.drop('Unnamed: 32',1,inplace=True)

data.head(4)

data.columns

data.dtypes

data.isnull().sum()

#Lets break the data into training and test sets

train,test=train_test_split(data,train_size=0.95,test_size=0.05)
train.size,test.size

sns.heatmap(train.corr(),xticklabels=False,yticklabels=False)

# We observe that a lot of features are related to each other. 

train.head(1)

train_labels=train['diagnosis']
test_labels=test['diagnosis']

#We drop the id and diagnosis columns in train and test

train.drop(['id','diagnosis'],1,inplace=True)
test.drop(['id','diagnosis'],1,inplace=True)

#We run a basic random forest on the data to know the feature importances.

forest=ensemble.RandomForestClassifier(n_estimators=100,n_jobs=-1)
forest.fit(train,train_labels)

importances=forest.feature_importances_
indices=np.argsort(importances)[::-1]
feature=train.columns
plt.yticks(range(len(indices)),feature[indices],fontsize=10)
plt.barh(range(len(indices)),importances[indices])
plt.tight_layout()

# We will rearrange the columns based on the feature importances

train2=train.ix[:,feature[indices]]
test2=test.ix[:,feature[indices]]

train2.head(2)

#Lets look at the overall distribution among features

train2.describe()

#We will need to normalize values because there is huge variation among the values. The area_mean has value 661 while
#the fractional_dimension_worst has mean value 0.083

scaler=StandardScaler(with_mean=True,with_std=True)
scaled_features=scaler.fit_transform(train2)
scaled_train_df=pd.DataFrame(scaled_features,index=train2.index,columns=train2.columns)

scaled_features2=scaler.fit_transform(test2)
scaled_test_df=pd.DataFrame(scaled_features2,index=test2.index,columns=test2.columns)

#We will join the labels

train3=scaled_train_df.join(train_labels)
test3=scaled_test_df.join(test_labels)

# We create training,validation and testing datasets

TRAIN,VAL=train_test_split(train3,train_size=0.8)
TEST=test3
x_TRAIN,y_TRAIN=TRAIN.drop('diagnosis',1),TRAIN['diagnosis']
x_VAL,y_VAL=VAL.drop('diagnosis',1),VAL['diagnosis']
x_TEST,y_TEST=TEST.drop('diagnosis',1),TEST['diagnosis']

#Logistic Regression

from sklearn.linear_model import LogisticRegression


logreg=LogisticRegression()
logreg.fit(x_TRAIN,y_TRAIN)
y_pred_logreg=logreg.predict(x_VAL)
val_accuracy_logreg=accuracy_score(y_pred_logreg,y_VAL)
val_accuracy_logreg

y_pred_logreg_=logreg.predict(x_TEST)
test_accuracy_logreg=accuracy_score(y_pred_logreg_,y_TEST)
test_accuracy_logreg

#Random Forest Classification

from sklearn.ensemble import RandomForestClassifier

forest=RandomForestClassifier(n_estimators=100)
forest.fit(x_TRAIN,y_TRAIN)
y_pred_forest=forest.predict(x_VAL)
val_accuracy_forest=accuracy_score(y_pred_forest,y_VAL)

y_pred_forest_=forest.predict(x_TEST)
test_accuracy_forest=accuracy_score(y_pred_forest_,y_TEST)
'validation accuracy= '+str(val_accuracy_forest)+'   '+'final accuracy= '+str(test_accuracy_forest)



