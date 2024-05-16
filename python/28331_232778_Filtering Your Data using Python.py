import swat

conn = swat.CAS(host, port, username, password)

cars = conn.read_csv('https://raw.githubusercontent.com/sassoftware/sas-viya-programming/master/data/cars.csv')
cars

len(cars)

cars.table.columninfo()

cars.where = 'Type = "Sports"'
cars

len(cars)

cars.head()

del cars.where
cars

cars.query('Type = "Sports"').head()

cars.query('Type = "Sports"', inplace=True)
cars

del cars.where
cars

cars[cars.Type == 'Sports'].head()

cars.Type == 'Sports'

(cars.Type == 'Sports').head(40)

cars[cars.Type == 'Sports'].head()

cars[(cars.Type == 'Sports') & (cars.Cylinders > 6)].head()

cars[cars.Type == 'Sports'][cars.Cylinders > 6].head()

sports8 = cars[cars.Type == 'Sports'][cars.Cylinders > 6]
sports8

conn.close()



