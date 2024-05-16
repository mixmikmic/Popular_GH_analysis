## perform imports and set-up
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from matplotlib import pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import scale
from scipy import stats

get_ipython().magic('matplotlib inline')
plt.style.use('ggplot') # emulate pretty r-style plots

# print numpy arrays with precision 4
np.set_printoptions(precision=4)

df = pd.read_csv('../../../data/Weekly.csv')
print('Weekly dataframe shape =', df.shape)
df.head()

# Compute correlation coeffecient matrix
correlations = df.corr(method='pearson')
print(correlations)

# Plot the Trading Volume vs. Year
fig, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=(18,4));

ax1.scatter(df.Year.values,df.Volume.values, facecolors='none', edgecolors='b');
ax1.set_xlabel('Year');
ax1.set_ylabel('Volume in Billions');

# Plot Lag1 vs Today's return
ax2.scatter(df.Lag1.values, df.Today.values, facecolors='none', edgecolors='b' );
ax2.set_xlabel('Lag1 Percent Return');
ax2.set_ylabel('Today\'s Percent Return');

# Plot Lag1 vs Today's return
ax3.scatter(df.Lag2.values, df.Today.values, facecolors='none', edgecolors='b' );
ax3.set_xlabel('Lag2 Percent Return');
ax3.set_ylabel('Today\'s Percent Return');

# Construct Design Matrix #
###########################
predictors = df.columns[1:7] # the lags and volume
X = sm.add_constant(df[predictors])

# Convert the Direction to Binary #
###################################
y = np.array([1 if el=='Up' else 0 for el in df.Direction.values])

# Construct the logit model #
###########################
logit = sm.Logit(y,X)
results=logit.fit()
print(results.summary())

# Get the predicted results for the full dataset
y_predicted = results.predict(X)
#conver the predicted probabilities to a class
y_predicted= np.array(y_predicted > 0.5, dtype=float)

# Build Confusion Matrix #
##########################
table = np.histogram2d(y_predicted, y, bins=2)[0]
print(pd.DataFrame(table, ['Down', 'Up'], ['Down', 'Up']))
print('\n')
print('Error Rate =', 1-(table[0,0]+table[1,1])/np.sum(table))

# Split Data #
##############
# get the Lag2 values for years less than =  2008
X_train = sm.add_constant(df[df.Year <= 2008].Lag2)
response_train = df[df.Year <= 2008].Direction
# convert responses to 0,1's
y_train = np.array([1 if el=='Up' else 0 for el in response_train])

# for the test set use the years > 2008
X_test = sm.add_constant(df[df.Year > 2008].Lag2)
response_test = df[df.Year > 2008].Direction
y_test = np.array([1 if el=='Up' else 0 for el in response_test])

# Construct Classifier and Fit #
################################
logit = sm.Logit(y_train, X_train)
results = logit.fit()
print(results.summary())
print('\n')

# Predict Test Set Responses #
##############################
y_predicted = results.predict(X_test)
#conver the predicted probabilities to a class
y_predicted= np.array(y_predicted > 0.5, dtype=float)

# Build Confusion Matrix #
##########################
table = np.histogram2d(y_predicted, y_test, bins=2)[0]
print('CONFUSION MATRIX')
print(pd.DataFrame(table, ['Down', 'Up'], ['Down', 'Up']))
print('\n')
print('Error Rate =', 1-(table[0,0]+table[1,1])/np.sum(table))

# Create LDA Classifier and Fit #
#################################
clf = LDA(solver='lsqr', store_covariance=True)
# No constant needed for LDA so reset the X_train
X_train = df[df.Year <= 2008].Lag2.values
# reshape so indexed by two indices
X_train = X_train.reshape((len(X_train),1))

# also go ahead and get test set and responses
X_test = df[df.Year > 2008].Lag2.values
# reshape into so indexed by two indices
X_test = X_test.reshape((len(X_test),1))

clf.fit(X_train, y_train)
print('Priors = ', clf.priors_ )
print('Class Means = ', clf.means_[0], clf.means_[1])
print('Coeffecients = ', clf.coef_)
print('\n')

# Predict Test Set Responses #
##############################
y_predicted = clf.predict(X_test)
#conver the predicted probabilities to class 0 or 1
y_predicted= np.array(y_predicted > 0.5, dtype=float)

# Build Confusion Matrix #
##########################
table = np.histogram2d(y_predicted, y_test, bins=2)[0]
print('CONFUSION MATRIX')
print(pd.DataFrame(table, ['Down', 'Up'], ['Down', 'Up']))
print('\n')
print('Error Rate =', 1-(table[0,0]+table[1,1])/np.sum(table))

# Build Classifier and Fit #
############################
qclf = QDA(store_covariances=True)
qclf.fit(X_train,y_train)

print('Priors = ', qclf.priors_ )
print('Class Means = ', qclf.means_[0], qclf.means_[1])
print('Covariances = ', qclf.covariances_)
print('\n')

# Predict Test Set Responses #
##############################
y_predict = qclf.predict(X_test)
#conver the predicted probabilities to class 0 or 1
y_predicted= np.array(y_predict > 0.5, dtype=float)

# Build Confusion Matrix #
##########################
table = np.histogram2d(y_predict, y_test, bins=2)[0]
print('CONFUSION MATRIX')
print(pd.DataFrame(table, ['Down', 'Up'], ['Down', 'Up']))
print('\n')
print('Error Rate =', 1-(table[0,0]+table[1,1])/np.sum(table))

# Build KNN Classifier and Fit #
################################
clf = KNeighborsClassifier(n_neighbors=1)
clf.fit(X_train, y_train)

# Predict Test Set Responses #
##############################
y_predicted = clf.predict(X_test)

table = np.histogram2d(y_predicted, y_test , bins=2)[0]
print(pd.DataFrame(table, ['Down', 'Up'], ['Down', 'Up']))
print('')
print('Error Rate =', 1-(table[0,0]+table[1,1])/np.sum(table))

# Build KNN Classifier and Fit #
################################
clf = KNeighborsClassifier(n_neighbors=20)
clf.fit(X_train, y_train)

# Predict Test Set Responses #
##############################
y_predicted = clf.predict(X_test)

table = np.histogram2d(y_predicted, y_test , bins=2)[0]
print(pd.DataFrame(table, ['Down', 'Up'], ['Down', 'Up']))
print('')
print('Error Rate =', 1-(table[0,0]+table[1,1])/np.sum(table))

# Split Data #
##############
predictors = df.columns[1:4]
X_train = sm.add_constant(df[df.Year <= 2008][predictors])
response_train = df[df.Year <= 2008].Direction
# convert responses to 0,1's
y_train = np.array([1 if el=='Up' else 0 for el in response_train])

# for the test set use the years > 2008
X_test = sm.add_constant(df[df.Year > 2008][predictors])
response_test = df[df.Year > 2008].Direction
y_test = np.array([1 if el=='Up' else 0 for el in response_test])

# Construct Classifier and Fit #
################################
logit = sm.Logit(y_train, X_train)
results = logit.fit()
print(results.summary())
print('\n')

# Predict Test Set Responses #
##############################
y_predicted = results.predict(X_test)
#conver the predicted probabilities to a class
y_predicted= np.array(y_predicted > 0.5, dtype=float)

# Build Confusion Matrix #
##########################
table = np.histogram2d(y_predicted, y_test, bins=2)[0]
print('CONFUSION MATRIX')
print(pd.DataFrame(table, ['Down', 'Up'], ['Down', 'Up']))
print('\n')
print('Error Rate =', 1-(table[0,0]+table[1,1])/np.sum(table))

# Add Interaction #
###################
# add the interaction term to the dataframe
df['Lag1xLag2'] = pd.Series(df.Lag1*df.Lag2, index=df.index)
predictors = ['Lag1', 'Lag2', 'Lag1xLag2']

# Split Data #
##############
X_train = sm.add_constant(df[df.Year <= 2008][predictors])
response_train = df[df.Year <= 2008].Direction
# convert responses to 0,1's
y_train = np.array([1 if el=='Up' else 0 for el in response_train])

# for the test set use the years > 2008
X_test = sm.add_constant(df[df.Year > 2008][predictors])
response_test = df[df.Year > 2008].Direction
y_test = np.array([1 if el=='Up' else 0 for el in response_test])

# Construct Classifier and Fit #
################################
logit = sm.Logit(y_train, X_train)
results = logit.fit()
print(results.summary())
print('\n')

# Predict Test Set Responses #
##############################
y_predicted = results.predict(X_test)
#conver the predicted probabilities to a class
y_predicted= np.array(y_predicted > 0.5, dtype=float)

# Build Confusion Matrix #
##########################
table = np.histogram2d(y_predicted, y_test, bins=2)[0]
print('CONFUSION MATRIX')
print(pd.DataFrame(table, ['Down', 'Up'], ['Down', 'Up']))
print('\n')
print('Error Rate =', 1-(table[0,0]+table[1,1])/np.sum(table))



