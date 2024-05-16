import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt


get_ipython().magic('matplotlib inline')


maindf = pd.read_csv('dataset/titanic/train.csv')

maindf.head()

maindf.tail()

maindf.describe()

maindf.describe(include=['object'])

maindf.columns

conditionsurvive = (maindf.Survived==1)

maindf.loc[[0,1,2],['PassengerId','Survived','Pclass']]

# Finding survive
maindf.loc[conditionsurvive,:].shape

# Finding non survive
maindf.loc[~conditionsurvive,:].shape

# Or anotther way is to group them by survived
maindf.Survived.value_counts()

maindf.groupby('Survived').count().Name

# Let's now group the data to find if most of them are male or female
maindf.groupby(['Survived','Sex']).count().Name

# Let's plot them out and see the result
maindf.groupby(['Survived','Sex']).count().Name.plot(kind='bar')

survivedsex = maindf.groupby(['Survived','Sex']).count().Name.reset_index()
maindfsurvivedis0 = survivedsex.loc[(survivedsex.Survived==0),:]
maindfsurvivedis1 = survivedsex.loc[(survivedsex.Survived==1),:]

maindfsurvivedis0

maindf.dtypes

maindf.info()

# Replace null in age with the average
maindf['Age'] = maindf.loc[:,['Age']].fillna(maindf['Age'].mean())

# Describing columns for analysis
# Drop useless columns such as passengerid ,name, (ticket, fare) can be described by socioeconomic status, and cabin (too much null)
cleandf = maindf.loc[:,['Survived','Pclass','Sex','Age','SibSp','Parch','Embarked']]

# We can transform Pclass and Embarked
# 1st ~ Upper; 2nd ~ Middle; 3rd ~ Lower
cleandf['socioeconomicstatus']=cleandf.Pclass.map({1:'upper',2:'middle',3:'lower'})

# (C = Cherbourg; Q = Queenstown; S = Southampton)
cleandf['embarkedport']=cleandf.Embarked.map({'C':'Cherbourg','Q':'Queenstown','S':'Southampton'})

# Dropping the used columns
cleandf.drop(['Pclass','Embarked'],axis=1,inplace=True)

# Group age for comparisons
cleandf.Age.hist()

# Let us try to separate this into ages
agesplit = [0,10,18,25,40,90]
agestatus = ['Adolescent','Teenager','Young Adult','Adult','Elder']

cleandf['agegroup']=pd.cut(cleandf.Age,agesplit,labels=agestatus)

# Create a feature where we count both numbers of siblings and parents
cleandf['familymembers']=cleandf.SibSp+cleandf.Parch

# Let us try to find whether the passengers are alone or not
hasfamily = (cleandf.familymembers>0)*1
cleandf['hasfamily'] = hasfamily

# Dropping the used columns
cleandf.drop(['SibSp','Parch','Age'],axis=1,inplace=True)

# Final transformed data
cleandf.head()

cleandf.to_csv('cleanedandtransformedtitanicdata.csv')

# Reading from csv
cleandf = pd.read_csv('cleanedandtransformedtitanicdata.csv')

cleandf.agegroup.value_counts().plot(kind='bar')

# The proportion of survivor with family
cleandf.groupby(['Survived','hasfamily']).count().agegroup.plot(kind='bar')

survived = pd.crosstab(index=cleandf.Survived, columns = cleandf.socioeconomicstatus,margins=True)
survived.columns = ['lower','middle','upper','rowtotal']
survived.index = ['died','survived','coltotal']

survived

survived/survived.ix['coltotal','rowtotal']

# Most of the lower class died, middle have same chances of survival, and upper are morelikely to survive

cleandf.groupby(['Survived','embarkedport']).count().agegroup

# create a figure with two subplots
fig = plt.figure(figsize=(10,10))
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)

notsurvivors = cleandf[cleandf.Survived==0].embarkedport.value_counts()
survivors= cleandf[cleandf.Survived==1].embarkedport.value_counts()


# plot each pie chart in a separate subplot
ax1.pie(notsurvivors,labels=notsurvivors.index);
ax2.pie(survivors,labels=survivors.index);

print(cleandf.socioeconomicstatus.value_counts())
cleandf.socioeconomicstatus.value_counts().plot(kind='bar')

isadult = cleandf.agegroup=='Adult'
issurvived = cleandf.Survived==1
isnotsurvived = cleandf.Survived==0

all = cleandf[isadult].groupby(['Sex','socioeconomicstatus']).count().Survived
survived = cleandf[isadult&issurvived].groupby(['Sex','socioeconomicstatus']).count().Survived
notsurvived = cleandf[isadult&isnotsurvived].groupby(['Sex','socioeconomicstatus']).count().Survived

survivedcrosstab = pd.crosstab(index=cleandf.Survived, columns = cleandf.socioeconomicstatus,margins=True)
survivedcrosstab.columns = ['lower','middle','upper','rowtotal']
survivedcrosstab.index = ['died','survived','coltotal']

survivedcrosstab

survivedcrosstab/survivedcrosstab.ix['coltotal','rowtotal']

survivedcrosstabsex = pd.crosstab(index=cleandf.Survived, columns = [cleandf['socioeconomicstatus'],cleandf['Sex']],margins=True)

survivedcrosstabsex

# Probability of survival
(survived/all).plot(kind='bar')

#dropping left and sales X for the df, y for the left
X = cleandf.drop(['Survived'],axis=1)
y = cleandf['Survived']

# Clean up x by getting the dummies
X=pd.get_dummies(X)

import numpy as np
from sklearn import preprocessing,cross_validation,neighbors,svm
#splitting the train and test sets
X_train, X_test, y_train,y_test= cross_validation.train_test_split(X,y,test_size=0.2)

from sklearn import tree
clftree = tree.DecisionTreeClassifier(max_depth=3)
clftree.fit(X_train,y_train)

# Visualizing the decision tree
from sklearn import tree
from scipy import misc
import pydotplus
import graphviz

def show_tree(decisionTree, file_path):
    tree.export_graphviz(decisionTree, out_file='tree.dot',feature_names=X_train.columns)
    graph = pydotplus.graphviz.graph_from_dot_file('tree.dot')
    graph.write_png('tree.png')
    i = misc.imread(file_path)
    
    fig, ax = plt.subplots(figsize=(18, 10))    
    ax.imshow(i, aspect='auto')

# To use it
show_tree(clftree, 'tree.png')

# Finding the accuracy of decision tree

from sklearn.metrics import accuracy_score, log_loss

print('****Results****')
train_predictions = clftree.predict(X_test)
acc = accuracy_score(y_test, train_predictions)
print("Accuracy: {:.4%}".format(acc))

