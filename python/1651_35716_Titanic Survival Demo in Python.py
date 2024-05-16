import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import random
import numpy as np
import pandas as pd
from sklearn import datasets, svm, cross_validation, tree, preprocessing, metrics
import sklearn.ensemble as ske

titanic_df = pd.read_csv('/Users/avkashchauhan/learn/seattle-workshop/titanic_list.csv')

titanic_df.describe

titanic_df.shape

titanic_df.columns

titanic_df.head()

titanic_df['survived'].mean()

titanic_df.groupby('pclass').mean()

class_sex_grouping = titanic_df.groupby(['pclass','sex']).mean()
class_sex_grouping

class_sex_grouping['survived'].plot.bar()

group_by_age = pd.cut(titanic_df["age"], np.arange(0, 90, 10))
age_grouping = titanic_df.groupby(group_by_age).mean()
age_grouping['survived'].plot.bar()

print "You can see the data set has lots of missing entities"
titanic_df.count()

# Fixing inconsistencies 
titanic_df["home.dest"] = titanic_df["home.dest"].fillna("NA")
#removing body, cabin and boat features
titanic_df = titanic_df.drop(['body','cabin','boat'], axis=1)
#removing all NA values
titanic_df = titanic_df.dropna()

print "You will see the values are consitant now"
titanic_df.count()

# We can also drop 'name','ticket','home.dest' features as it will not help
titanic_df = titanic_df.drop(['name','ticket','home.dest'], axis=1)
titanic_df.count()

titanic_df.sex = preprocessing.LabelEncoder().fit_transform(titanic_df.sex)
titanic_df.sex
# Now SEX convers to 0 and 1 instead of male or female 

titanic_df.embarked = preprocessing.LabelEncoder().fit_transform(titanic_df.embarked)
titanic_df.embarked

# Create a dataframe which has all features we will use for model building
X = titanic_df.drop(['survived'], axis=1).values

y = titanic_df['survived'].values

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.2)

#Decision Tree Classifier
classify_dt = tree.DecisionTreeClassifier(max_depth=10)

print " This result means the model correctly predicted survival rate of given value %"
classify_dt.fit (X_train, y_train)
scr = classify_dt.score (X_test, y_test)
print "score : " , scr
print "Model is able to correctly predict survival rate of", scr *100 , "% time.."

# Creating a vlidator data which works on 80%-20% 
shuffle_validator = cross_validation.ShuffleSplit(len(X), n_iter=20, test_size=0.2, random_state=0)

def test_classifier(clf):
    scores = cross_validation.cross_val_score(clf, X, y, cv=shuffle_validator)
    print("Accuracy: %0.4f (+/- %0.2f)" % (scores.mean(), scores.std()))

test_classifier(classify_dt)
# Note: If you run shuffle_validator again and then run test classifier, you will see different accuracy

clf_rf = ske.RandomForestClassifier(n_estimators=50)
test_classifier(clf_rf)

# Performing Prediction

clf_rf.fit(X_train, y_train)
clf_rf.score(X_test, y_test)

clf_gb = ske.GradientBoostingClassifier(n_estimators=50)
test_classifier(clf_gb)

# Performing Prediction

clf_gb.fit(X_train, y_train)
clf_gb.score(X_test, y_test)

eclf = ske.VotingClassifier([('dt', classify_dt), ('rf', clf_rf), ('gb', clf_gb)])
test_classifier(eclf)

# Performing Prediction

eclf.fit(X_train, y_train)
eclf.score(X_test, y_test)

# Collection 10 records from each passenger class - Create datset of 30 records
passengers_set_1 = titanic_df[titanic_df.pclass == 1].iloc[:10,:].copy()
passengers_set_2 = titanic_df[titanic_df.pclass == 2].iloc[:10,:].copy()
passengers_set_3 = titanic_df[titanic_df.pclass == 3].iloc[:10,:].copy()
passenger_set = pd.concat([passengers_set_1,passengers_set_2,passengers_set_3])
#testing_set = preprocess_titanic_df(passenger_set)

passenger_set.count()
# You must see 30 uniform records

passenger_set.survived.count()

titanic_df.count()

passenger_set_new = passenger_set.drop(['survived'], axis=1)
prediction = clf_rf.predict(passenger_set_new)

passenger_set[passenger_set.survived != prediction]



