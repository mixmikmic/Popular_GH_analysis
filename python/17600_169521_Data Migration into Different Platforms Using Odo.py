from odo import odo
import pandas as pd

csv_dataset = "/Users/RichardAfolabi/myGitHub/Data_Science_Harvard/2014_data/countries.csv"

# Convert from CSV to Dataframe
df = odo(csv_dataset, pd.DataFrame)
df.head()

# Convert from CSV to List
odo(csv_dataset, list)[:5]

# Convert from dataframe to JSON and dump in directory
# odo(df, 'json_dataset.json')

# Use Spark SQL Context to read json file
sp_df = sqlContext.read.json('json_dataset.json').take(5)
sp_df

# Create Spark SchemaRDD using Odo
data = [('Alice', 300.0), ('Bob', 200.0), ('Donatello', -100.0)]
# odo(data, sqlContext.sql, dshape='var * {Country: string, Region: string}')



