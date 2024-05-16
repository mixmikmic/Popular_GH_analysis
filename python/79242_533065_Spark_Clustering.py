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

# Spin up a local Spark Session (with 4 executors)
spark = SparkSession.builder.master("local[4]").appName('my_awesome').getOrCreate()

# Have Spark read in the Parquet File
spark_df = spark.read.parquet("dns.parquet")

# Get information about the Spark DataFrame
num_rows = spark_df.count()
print("Number of Rows: {:d}".format(num_rows))
columns = spark_df.columns
print("Columns: {:s}".format(','.join(columns)))

spark_df.groupby('qtype_name','proto').count().sort('count', ascending=False).show()

# Add a column with the string length of the DNS query
from pyspark.sql.functions import col, length

# Create new dataframe that includes two new column
spark_df = spark_df.withColumn('query_length', length(col('query')))
spark_df = spark_df.withColumn('answer_length', length(col('answers')))

# Plotting defaults
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
from bat.utils import plot_utils
plot_utils.plot_defaults()

# Show histogram of the Spark DF request body lengths
bins, counts = spark_df.select('query_length').rdd.flatMap(lambda x: x).histogram(50)

# This is a bit awkward but I believe this is the correct way to do it
plt.hist(bins[:-1], bins=bins, weights=counts, log=True)
plt.grid(True)
plt.xlabel('DNS Query Lengths')
plt.ylabel('Counts')

# Show histogram of the Spark DF request body lengths
bins, counts = spark_df.select('answer_length').rdd.flatMap(lambda x: x).histogram(50)

# This is a bit awkward but I believe this is the correct way to do it
plt.hist(bins[:-1], bins=bins, weights=counts, log=True)
plt.grid(True)
plt.xlabel('DNS Answer Lengths')
plt.ylabel('Counts')

from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler

categoricalColumns = ['qtype_name', 'proto']
stages = [] 

for categoricalCol in categoricalColumns:
    stringIndexer = StringIndexer(inputCol=categoricalCol, 
        outputCol=categoricalCol+"Index")
    encoder = OneHotEncoder(inputCol=categoricalCol+"Index", 
        outputCol=categoricalCol+"classVec")
    stages += [stringIndexer, encoder]

numericCols = ['query_length', 'answer_length', 'Z', 'rejected']
assemblerInputs = [c + "classVec" for c in categoricalColumns] + numericCols
assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")
stages += [assembler]

pipeline = Pipeline(stages=stages)
pipelineModel = pipeline.fit(spark_df)
spark_df = pipelineModel.transform(spark_df)

spark_df.select('features').show()

from pyspark.ml.clustering import KMeans

# Trains a k-means model.
kmeans = KMeans().setK(70)
model = kmeans.fit(spark_df)

# Evaluate clustering by computing Within Set Sum of Squared Errors.
wssse = model.computeCost(spark_df)
print("Within Set Sum of Squared Errors = " + str(wssse))

# Shows the result.
centers = model.clusterCenters()
print("Cluster Centers: ")
for center in centers:
    print(center)

features = ['qtype_name', 'proto', 'query_length', 'answer_length', 'Z', 'rejected']
transformed = model.transform(spark_df).select(features + ['prediction'])
transformed.collect()
transformed.show()

transformed.groupby(features + ['prediction']).count().sort('prediction').show(50)

