# Third Party Imports
import pyspark
from pyspark.sql import SparkSession
import pyarrow

# Local imports
import bat
from bat.log_to_parquet import log_to_parquet

# Good to print out versions of stuff
print('BAT: {:s}'.format(bat.__version__))
print('PySpark: {:s}'.format(pyspark.__version__))
print('PyArrow: {:s}'.format(pyarrow.__version__))

# Create a Parquet file from a Bro Log with a super nice BAT method.
log_to_parquet('/Users/briford/data/bro/sec_repo/http.log', 'http.parquet')

# Spin up a local Spark Session (with 4 executors)
spark = SparkSession.builder.master('local[4]').appName('my_awesome').getOrCreate()

# Have Spark read in the Parquet File
get_ipython().magic('time spark_df = spark.read.parquet("http.parquet")')

spark_df.rdd.getNumPartitions()

# Get information about the Spark DataFrame
num_rows = spark_df.count()
print("Number of Rows: {:d}".format(num_rows))
columns = spark_df.columns
print("Columns: {:s}".format(','.join(columns)))

spark_df.select(['`id.orig_h`', 'host', 'uri', 'status_code', 'user_agent']).show(5)

get_ipython().magic("time spark_df.groupby('method','status_code').count().sort('count', ascending=False).show()")

