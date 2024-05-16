import swat

conn = swat.CAS(host, port, username, password)

cars = conn.read_csv('https://raw.githubusercontent.com/sassoftware/sas-viya-programming/master/data/cars.csv')
cars

cars.fetch(to=5)

cars.fetch(to=5, sortby=['Cylinders', 'EngineSize'])

cars.fetch(to=5, sortby=['Cylinders', dict(name='EngineSize', order='descending')])

sorted_cars = cars.sort_values(['Cylinders', 'EngineSize'])
sorted_cars.fetch(to=5)

sorted_cars.head()

cars.head()

cars.sort_values(['Cylinders', 'EngineSize'], ascending=[True, False], inplace=True)
cars.head()

conn.close()



