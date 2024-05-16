from pyspark.sql import SparkSession
from bat import log_to_dataframe
import pandas as pd

# Convert Bro log to Pandas DataFrame
dns_df = log_to_dataframe.LogToDataFrame('../data/dns.log')
dns_df.head()

# Spin up a local Spark Session (with 4 executors)
spark = SparkSession.builder.master('local[4]').appName('my_awesome').getOrCreate()

# Convert to Spark DF
spark_df = spark.createDataFrame(dns_df)

# Some simple spark operations
num_rows = spark_df.count()
print("Number of Spark DataFrame rows: {:d}".format(num_rows))
columns = spark_df.columns
print("Columns: {:s}".format(','.join(columns)))

# Some simple spark operations
spark_df.groupBy('proto').count().show()

# Some simple spark operations
spark_df.groupBy('`id.orig_h`', '`id.resp_h`').count().show()

# Add a column with the string length of the DNS query
from pyspark.sql.functions import col, length

# Create new dataframe that includes new column
spark_df = spark_df.withColumn('query_length', length(col('query')))
spark_df[['query', 'query_length']].show()

# Plotting defaults
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
from bat.utils import plot_utils
plot_utils.plot_defaults()

# Show histogram of the Spark DF query lengths
bins, counts = spark_df.select('query_length').rdd.flatMap(lambda x: x).histogram(20)

# This is a bit awkward but I believe this is the correct way to do it
plt.hist(bins[:-1], bins=bins, weights=counts)
plt.grid(True)
plt.xlabel('Query Lengths')
plt.ylabel('Counts')

# Compare the computation of query_length and resulting histogram with Pandas
dns_df['query_length'] = dns_df['query'].str.len()
dns_df['query_length'].hist(bins=20)
plt.xlabel('Query Lengths')
plt.ylabel('Counts')

