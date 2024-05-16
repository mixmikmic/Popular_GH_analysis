import pandas as pd
training_data = pd.read_csv('../training_data.csv')

X = training_data.drop(['Formation', 'Well Name', 'Depth','Facies'], axis=1).values
y = training_data['Facies'].values

wells = training_data["Well Name"].values

from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneGroupOut

logo = LeaveOneGroupOut()

for train, test in logo.split(X, y, groups=wells):
    well_name = wells[test[0]]
    score = SVC().fit(X[train], y[train]).score(X[test], y[test])
    print("{:>20s}  {:.3f}".format(well_name, score))



