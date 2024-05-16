import pandas as pd
import numpy as np

data = pd.read_csv('data/train.csv')

data.head(4)

# let's use apply function to get the length of names
data["Name_length"] = data.Name.apply(len)

data.loc[0:5, ["Name", "Name_length"]]

# let's get the mean price on fare column
data["Fare_mean"] = data.Fare.apply(np.mean)

data.loc[0:5, ["Fare", "Fare_mean"]]

data.Name.str.split('.')[0][0].split(',')[1]

# let's get the name perfix, like Mr. Mrs. Mss. Ms...
data['prefix'] = data.Name.str.split('.').apply(lambda x: x[0].split(',')[1])

data.loc[0:10, ['Name', 'prefix']]

del data['dummy_prefix']

data.tail()

# let's get the unique prefix
data['prefix'].unique()

# let's use apply function to combined prefixes, 
# male = Mr Master, Don, rev, Dr, sir, col, capt, == 0
# female = Mrs miss, Mme, Ms, Lady, Mlle, the countess,Jonkheer  == 1

dummy_pre = data.groupby('prefix')

#list(data.groupby('prefix'))

dummy_pre.count()

get_dummy = data.prefix

pd.get_dummies(data['prefix'])

data.head()



