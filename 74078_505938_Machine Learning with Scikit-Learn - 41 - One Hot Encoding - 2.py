import pandas as pd
from IPython.display import display

data = pd.read_csv('adult.data', header=None, index_col=False, names=['age', 'workclass', 'fnlwgt', 'education', 
                                                                      'education-num', 'marital-status', 'occupation', 
                                                                      'relationship', 'race', 'gender', 'capital-gain', 
                                                                      'capital-loss', 'hours-per-week', 'native-country', 
                                                                      'income'])

data = data[['age', 'workclass', 'education', 'gender', 'hours-per-week', 'occupation', 'income']]
display(data)

print('Original Features:\n', list(data.columns), '\n')
data_dummies = pd.get_dummies(data)
print('Features after One-Hot Encoding:\n', list(data_dummies.columns))

features = data_dummies.ix[:, 'age':'occupation_ Transport-moving']
X = features.values
y = data_dummies['income_ >50K'].values

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

print('Logistic Regression score on the test set: {:.2f}'.format(logreg.score(X_test, y_test)))



