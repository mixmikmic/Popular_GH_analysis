import swat

conn = swat.CAS(host, port, username, password)

get_ipython().magic('pinfo conn.simple')

help(conn.simple)

cars = conn.read_csv('https://raw.githubusercontent.com/sassoftware/sas-viya-programming/master/data/cars.csv')
out = cars.summary()
out

df = out['Summary']
df.set_index(df.columns[0], inplace=True)
df

df.loc[['MSRP', 'Invoice'], ['Min', 'Mean', 'Max']]

cars.describe()

cars.min()

cars.max()

cars.std()

conn.close()



