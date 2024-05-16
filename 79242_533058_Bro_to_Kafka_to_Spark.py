import pyspark
from pyspark.sql import SparkSession

# Always good to print out versions of libraries
print('PySpark: {:s}'.format(pyspark.__version__))

# Spin up a local Spark Session
# Please note: the config line is an important bit,
# readStream.format('kafka') won't work without it
spark = SparkSession.builder.master('local[4]').appName('my_awesome')        .config('spark.jars.packages', 'org.apache.spark:spark-sql-kafka-0-10_2.11:2.2.0')        .getOrCreate()

# Optimize the conversion to Pandas
spark.conf.set("spark.sql.execution.arrow.enable", "true")

# SUBSCRIBE: Setup connection to Kafka Stream 
raw_data = spark.readStream.format('kafka')   .option('kafka.bootstrap.servers', 'localhost:9092')   .option('subscribe', 'dns')   .option('startingOffsets', 'latest')   .load()

# ETL: Hardcoded Schema for DNS records (do this better later)
from pyspark.sql.types import StructType, StringType, BooleanType, IntegerType
from pyspark.sql.functions import from_json, to_json, col, struct

dns_schema = StructType()     .add('ts', StringType())     .add('uid', StringType())     .add('id.orig_h', StringType())     .add('id.orig_p', IntegerType())     .add('id.resp_h', StringType())     .add('id.resp_p', IntegerType())     .add('proto', StringType())     .add('trans_id', IntegerType())     .add('query', StringType())     .add('qclass', IntegerType())     .add('qclass_name', StringType())     .add('qtype', IntegerType())     .add('qtype_name', StringType())     .add('rcode', IntegerType())     .add('rcode_name', StringType())     .add('AA', BooleanType())     .add('TC', BooleanType())     .add('RD', BooleanType())     .add('RA', BooleanType())     .add('Z', IntegerType())     .add('answers', StringType())     .add('TTLs', StringType())     .add('rejected', BooleanType())

# ETL: Convert raw data into parsed and proper typed data
parsed_data = raw_data   .select(from_json(col("value").cast("string"), dns_schema).alias('data'))   .select('data.*')

# FILTER/AGGREGATE: In this case a simple groupby operation
group_data = parsed_data.groupBy('`id.orig_h`', 'qtype_name').count()

# At any point in the pipeline you can see what you're getting out
group_data.printSchema()

# Take the end of our pipeline and pull it into memory
dns_count_memory_table = group_data.writeStream.format('memory')   .queryName('dns_counts')   .outputMode('complete')   .start()

dns_count_memory_table

# Create a Pandas Dataframe by querying the in memory table and converting
dns_counts_df = spark.sql("select * from dns_counts").toPandas()
print('DNS Query Counts = {:d}'.format(len(dns_counts_df)))
dns_counts_df.sort_values(ascending=False, by='count')

# Create a Pandas Dataframe by querying the in memory table and converting
dns_counts_df = spark.sql("select * from dns_counts").toPandas()
print('DNS Query Counts = {:d}'.format(len(dns_counts_df)))
dns_counts_df.sort_values(ascending=False, by='count')

# We should stop our streaming pipeline when we're done
dns_count_memory_table.stop()

