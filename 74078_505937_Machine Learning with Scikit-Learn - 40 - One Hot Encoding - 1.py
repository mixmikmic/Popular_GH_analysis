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





