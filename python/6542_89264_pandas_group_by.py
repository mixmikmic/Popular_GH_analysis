import pandas as pd
df = pd.read_csv("weather_by_cities.csv")
df

g = df.groupby("city")
g

for city, data in g:
    print("city:",city)
    print("\n")
    print("data:",data)    

g.get_group('mumbai')

g.max()

g.mean()

g.min()

g.describe()

g.size()

g.count()

get_ipython().magic('matplotlib inline')
g.plot()

