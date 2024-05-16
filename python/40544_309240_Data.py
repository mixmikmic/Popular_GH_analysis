import pandas
df = pandas.read_csv('../data/iris.csv')
df.head(20)

import json

with open('../data/Museums_in_DC.geojson') as f:
    s = json.loads(f.read())

s



