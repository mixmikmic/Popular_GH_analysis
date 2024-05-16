import sqlite3 as sl
import pandas as pd

query = "SELECT code, airport, city, state, latitude, longitude FROM airports LIMIT 10 ;"

database = '/home/data_scientist/rppdm/database/rppds'

with sl.connect(database) as con:
    data = pd.read_sql(query, con, index_col ='code')
    
    print(data)

query = "SELECT code, airport, city, state, latitude, longitude FROM airports LIMIT 100 ;"

with sl.connect(database) as con:
    data = pd.read_sql(query, con, index_col ='code')
    
    print(data[data.state == 'MS'])

# Creating table automatically works better if columns are explicitly listed.

query = "SELECT code, airport, city, state, latitude, longitude FROM airports ;"
with sl.connect(database) as con:
    data = pd.read_sql(query, con)

    data[data.state == 'IL'].to_sql('ILAirports', con)

with sl.connect(database) as con:
    data = pd.read_sql('SELECT code, city, airport, latitude, longitude FROM ILAirports', 
                       con, index_col ='code')
    
    print(data[10:20])

