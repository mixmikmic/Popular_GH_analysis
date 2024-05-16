import numpy as np
import pandas as pd
import scipy as sp
from sklearn import preprocessing
from sklearn.cross_validation import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.tree import DecisionTreeClassifier as DecisionTree
from sklearn.ensemble import RandomForestClassifier as RandomForest
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
import matplotlib
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

#Train

df = pd.read_csv('data/flu_train.csv')

df = df[~np.isnan(df['flu'])]

df.head()

#Clean and encode

encode = preprocessing.LabelEncoder()

for column in df.columns:
    if df[column].dtype == np.object:
        df[column] = df[column].fillna('')
        df.loc[:, column] = encode.fit_transform(df[column])
        
df = df.fillna(0)

df.head()

#Test

df_test = pd.read_csv('data/flu_test.csv')

df_test = df_test[~np.isnan(df_test['flu'])]

df_test.head()

#Clean and encode

encode = preprocessing.LabelEncoder()

for column in df_test.columns:
    if df_test[column].dtype == np.object:
        df_test[column] = df_test[column].fillna('')
        df_test.loc[:, column] = encode.fit_transform(df_test[column])
        
df_test = df_test.fillna(0)

df_test.head()

#What's up in each set

x = df.values[:, :-2]
y = df.values[:, -2]

x_test = df_test.values[:, :-2]
y_test = df_test.values[:, -2]

print('x train shape:', x.shape)
print('x test shape:', x_test.shape)
print('train class 0: {}, train class 1: {}'.format(len(y[y==0]), len(y[y==1])))
print('train class 0: {}, train class 1: {}'.format(len(y_test[y_test==0]), len(y_test[y_test==1])))

def expected_score(model, x_test, y_test):
    overall = 0
    class_0 = 0
    class_1 = 0
    for i in range(100):
        sample = np.random.choice(len(x_test), len(x_test))
        x_sub_test = x_test[sample]
        y_sub_test = y_test[sample]
        
        overall += model.score(x_sub_test, y_sub_test)
        class_0 += model.score(x_sub_test[y_sub_test==0], y_sub_test[y_sub_test==0])
        class_1 += model.score(x_sub_test[y_sub_test==1], y_sub_test[y_sub_test==1])

    return pd.Series([overall / 100., 
                      class_0 / 100.,
                      class_1 / 100.],
                      index=['overall accuracy', 'accuracy on class 0', 'accuracy on class 1'])

score = lambda model, x_test, y_test: pd.Series([model.score(x_test, y_test), 
                                                 model.score(x_test[y_test==0], y_test[y_test==0]),
                                                 model.score(x_test[y_test==1], y_test[y_test==1])], 
                                                index=['overall accuracy', 'accuracy on class 0', 'accuracy on class 1'])

#KNN
knn = KNN(n_neighbors=2)
knn.fit(x, y)

knn_scores = score(knn, x, y)
print('knn')

#Unweighted logistic regression
unweighted_logistic = LogisticRegression(C=1000)
unweighted_logistic.fit(x, y)

unweighted_log_scores = score(unweighted_logistic, x, y)
print('unweighted log')


#Weighted logistic regression
weighted_logistic = LogisticRegression(C=1000, class_weight='balanced')
weighted_logistic.fit(x, y)

weighted_log_scores = score(weighted_logistic, x, y)
print('weighted log')


#LDA
lda = LDA()
lda.fit(x, y)

lda_scores = score(lda, x, y)
print('lda')

#QDA
qda = QDA()
qda.fit(x, y)

qda_scores = score(qda, x, y)
print('qda')

#Decision Tree
tree = DecisionTree(max_depth=50, class_weight='balanced', criterion='entropy')
tree.fit(x, y)

tree_scores = score(tree, x, y)
print('tree')


#Random Forest
rf = RandomForest(class_weight='balanced')
rf.fit(x, y)

rf_scores = score(rf, x, y)

print('rf')

#SVC
svc = SVC(C=100, class_weight='balanced')
svc.fit(x, y)

svc_scores = score(svc, x, y)

print('svc')

#Score Dataframe
score_df = pd.DataFrame({'knn': knn_scores, 
                         'unweighted logistic': unweighted_log_scores,
                         'weighted logistic': weighted_log_scores,
                         'lda': lda_scores,
                         'qda': qda_scores,
                         'tree': tree_scores,
                         'rf': rf_scores, 
                         'svc': svc_scores})
score_df

Cs = 10.**np.arange(-3, 4, 1)
scores = []
for C in Cs:
    print('C:', C)
    weighted_log_scores = np.array([0., 0., 0.])
    kf = KFold(len(x), n_folds=10, shuffle=True, random_state=10)
    for train_index, test_index in kf:
        x_validate_train, x_validate_test = x[train_index], x[test_index]
        y_validate_train, y_validate_test = y[train_index], y[test_index]

        weighted_logistic = LogisticRegression(C=C, class_weight='balanced')
        weighted_logistic.fit(x_validate_train, y_validate_train)

        weighted_log_scores += score(weighted_logistic, x_validate_test, y_validate_test).values

    scores.append(weighted_log_scores / 10.)

scores = pd.DataFrame(np.array(scores).T, columns=[str(C) for C in Cs], index=['overall accuracy', 'accuracy on class 0', 'accuracy on class 1'])

scores

#Weighted logistic regression
weighted_logistic = LogisticRegression(C=100, class_weight='balanced')
weighted_logistic.fit(x, y)
weighted_log_scores = score(weighted_logistic, x_test, y_test)
weighted_log_scores

#--------  cost
# A function that computes the expected cost of the public healthy policy based on the 
# classifications generated by your model
# Input: 
#      y_true (true class labels: 0, 1, 2)
#      y_pred (predicted class labels: 0, 1, 2)
# Returns: 
#      total_cost (expected total cost)

def cost(y_true, y_pred):
    cost_of_treatment_1 = 29500
    cost_of_treatment_2 = 45000
    cost_of_intervention_1 = 4150
    cost_of_intervention_2 = 4250
    cost_of_vaccine = 15
    
    prob_complications_untreated = 0.65
    prob_complications_1 = 0.30
    prob_complications_2 = 0.15
    
    trials = 1000    
    
    intervention_cost = cost_of_intervention_1 * len(y_pred[y_pred==1]) + cost_of_intervention_2 * len(y_pred[y_pred==2])

    vaccine_cost = cost_of_vaccine * len(y_pred[y_pred==0])
    
    false_neg_1 = ((y_true == 1) & (y_pred == 2)).sum()
    false_neg_2 = ((y_true == 2) & (y_pred == 1)).sum()
    
    untreated_1 = ((y_true == 1) & (y_pred == 0)).sum()    
    untreated_2 = ((y_true == 2) & (y_pred == 0)).sum()
    
    false_neg_1_cost = np.random.binomial(1, prob_complications_1, (false_neg_1, trials)) * cost_of_treatment_1
    false_neg_2_cost = np.random.binomial(1, prob_complications_2, (false_neg_2, trials)) * cost_of_treatment_2
    untreated_1_cost = np.random.binomial(1, prob_complications_untreated, (untreated_1, trials)) * cost_of_treatment_1
    untreated_2_cost = np.random.binomial(1, prob_complications_untreated, (untreated_2, trials)) * cost_of_treatment_2
    
    false_neg_1_cost = false_neg_1_cost.sum(axis=0)
    expected_false_neg_1_cost = false_neg_1_cost.mean()
    
    false_neg_2_cost = false_neg_2_cost.sum(axis=0)
    expected_false_neg_2_cost = false_neg_2_cost.mean()
    
    untreated_1_cost = untreated_1_cost.sum(axis=0)
    expected_untreated_1_cost = untreated_1_cost.mean()
    
    untreated_2_cost = untreated_2_cost.sum(axis=0)
    expected_untreated_2_cost = untreated_2_cost.mean()
    
    total_cost = vaccine_cost + intervention_cost + expected_false_neg_1_cost + expected_false_neg_2_cost + expected_untreated_1_cost + expected_untreated_2_cost
    
    return total_cost

x = df.values[:, :-2]
y = df.values[:, -1]
y = y - 1

x_test = df_test.values[:, :-2]
y_test = df_test.values[:, -1]

y_test = y_test - 1

score = lambda model, x_test, y_test: pd.Series([model.score(x_test, y_test), 
                                                 model.score(x_test[y_test==0], y_test[y_test==0]),
                                                 model.score(x_test[y_test==1], y_test[y_test==1]), 
                                                 model.score(x_test[y_test==2], y_test[y_test==2]), 
                                                 cost(y_test, model.predict(x_test))],
                                                index=['overall accuracy', 'accuracy on class 0', 'accuracy on class 1', 'accuracy on class 2', 'total cost'])

Cs = 10.**np.arange(-3, 4, 1)
scores = []
for C in Cs:
    print('C:', C)
    weighted_log_scores = np.array([0., 0., 0., 0., 0.])
    kf = KFold(len(x), n_folds=10, shuffle=True, random_state=10)
    for train_index, test_index in kf:
        x_validate_train, x_validate_test = x[train_index], x[test_index]
        y_validate_train, y_validate_test = y[train_index], y[test_index]

        weighted_logistic = LogisticRegression(C=C, class_weight={0:0.7, 1:10, 2:10})
        weighted_logistic.fit(x_validate_train, y_validate_train)

        weighted_log_scores += score(weighted_logistic, x_validate_test, y_validate_test).values

    scores.append(weighted_log_scores / 10.)

scores = pd.DataFrame(np.array(scores).T, columns=[str(C) for C in Cs], index=['overall accuracy', 'accuracy on class 0', 'accuracy on class 1', 'accuracy on class 2', 'total cost'])

scores

#Weighted logistic regression
weighted_logistic = LogisticRegression(C=100, class_weight={0:0.7, 1:10, 2:10})
weighted_logistic.fit(x, y)
weighted_log_scores = score(weighted_logistic, x, y)
weighted_log_scores

#Weighted logistic regression
weighted_log_scores = score(weighted_logistic, x_test, y_test)
weighted_log_scores

print('minimimum cost on train:', cost(y, y))
print('minimimum cost on test:', cost(y_test, y_test))

print('simple model cost on train:', cost(y, np.array([0] * len(y))))
print('simple model cost on test:', cost(y_test, np.array([0] * len(y_test))))

print('simple model cost on train:', cost(y, np.array([1] * len(y))))
print('simple model cost on test:', cost(y_test, np.array([1] * len(y_test))))

print('simple model cost on train:', cost(y, np.array([2] * len(y))))
print('simple model cost on test:', cost(y_test, np.array([2] * len(y_test))))

