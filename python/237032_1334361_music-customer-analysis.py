import os
import pandas as pd
import numpy as np

from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession, SQLContext

from pyspark.sql.types import *
import pyspark.sql.functions as F
from pyspark.sql.functions import udf, col

# Visualization
import seaborn as sns
import matplotlib.pyplot as plt

# Visualization
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

pd.set_option('display.max_columns', 50)
pd.set_option('display.max_colwidth', 50)

from matplotlib import rcParams
sns.set(context='notebook', style='whitegrid', rc={'figure.figsize': (6,4)})
rcParams['figure.figsize'] = 6,4

# this allows plots to appear directly in the notebook
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")

# setting random seed for notebook reproducability
rnd_seed=42
np.random.seed=rnd_seed
np.random.set_state=rnd_seed

os.environ['SPARK_HOME']

spark = (SparkSession
         .builder
         .master("local[*]")
         .appName("music-customer-analysis")
         .getOrCreate())

spark

sc = spark.sparkContext
sc

sqlContext = SQLContext(spark.sparkContext)
sqlContext

MUSIC_TRACKS_DATA = 'data/tracks.csv'
CUSTOMER_DATA =     'data/cust.csv'

# define the schema, corresponding to a line in the csv data file for music
music_schema = StructType([
    StructField('event_id', IntegerType(), nullable=True),
    StructField('customer_id', IntegerType(), nullable=True),
    StructField('track_id', StringType(), nullable=True),
    StructField('datetime', StringType(), nullable=True),
    StructField('is_mobile', IntegerType(), nullable=True),
    StructField('zip', IntegerType(), nullable=True)]
  )

# define the schema, corresponding to a line in the csv data file for customer
cust_schema = StructType([
    StructField('customer_id', IntegerType(), nullable=True),
    StructField('name', StringType(), nullable=True),
    StructField('gender', IntegerType(), nullable=True),
    StructField('address', StringType(), nullable=True),
    StructField('zip', IntegerType(), nullable=True),
    StructField('sign_date', StringType(), nullable=True),
    StructField('status', IntegerType(), nullable=True),
    StructField('level', IntegerType(), nullable=True),
    StructField('campaign', IntegerType(), nullable=True),
    StructField('lnkd_with_apps', IntegerType(), nullable=True)]
  )

# Load data
music_df = spark.read.csv(path=MUSIC_TRACKS_DATA, schema=music_schema).cache()
cust_df = spark.read.csv(path=CUSTOMER_DATA, schema=cust_schema, header=True).cache()

# How many music data rows
music_df.count()

music_df.show(5)

# How many customer data rows
cust_df.count()

cust_df.show(5)

music_df = music_df.withColumn('hour', F.hour('datetime')).cache()

music_df.show(5)

music_df = (music_df
    .withColumn('night', F.when((col('hour') < 5) | (col('hour') == 23), 1).otherwise(0))
    .withColumn('morn', F.when((col('hour') >= 5) & (col('hour') < 12), 1).otherwise(0))
    .withColumn('aft', F.when((col('hour') >= 12) & (col('hour') < 17), 1).otherwise(0))
    .withColumn('eve', F.when((col('hour') >= 17) & (col('hour') < 22), 1).otherwise(0)))

music_df.show(5)

cust_profile_df = (music_df.select(['customer_id', 'track_id', 'night', 'morn', 'aft', 'eve', 'is_mobile'])
     .groupBy('customer_id')
     .agg(F.countDistinct('track_id'), F.sum('night'), F.sum('morn'), F.sum('aft'), F.sum('eve'), F.sum('is_mobile'))).cache()

cust_profile_df.show(10)

cust_profile_df.select([c for c in cust_profile_df.columns if c not in ['customer_id']]).describe().show()

cust_df.show(5)

# Map from level number to actual level string
level_map = {0:"Free", 1:"Silver", 2:"Gold"}

# Define a udf
udfIndexTolevel = udf(lambda x: level_map[x], StringType())

result_df = (cust_df.join(cust_profile_df, on='customer_id', how='inner')
                     .select([udfIndexTolevel('level').alias('level'), 'sum(night)', 'sum(morn)', 'sum(aft)', 'sum(eve)'])
                     .groupBy('level')
                     .agg(F.avg('sum(aft)').alias('Afternoon'), 
                          F.avg('sum(eve)').alias('Evening'), 
                          F.avg('sum(morn)').alias('Morning'), 
                          F.avg('sum(night)').alias("Night")))

result_df.cache().show()

result_df.toPandas().plot.bar(x='level', figsize=(12, 4));

result_df.unpersist()

result_df = (cust_df.select(['level', (F.when(col('gender') == 0, "Male").otherwise("Female")).alias('gender')])
                 .groupBy('level')
                 .pivot('gender')
                 .count()
                 .orderBy('level', ascending=False))

result_df.cache().show()

result_df.toPandas().set_index('level').plot.barh(stacked=True);

result_df.unpersist()

result_df = cust_df.groupBy('zip').count().orderBy('count', ascending=False).limit(10)

result_df.cache().show()

result_df.toPandas().plot.barh(x='zip');

result_df.unpersist()

# Map from campaign number to actual campaign string
campaign_map = {0:"None", 1:"30DaysFree", 2:"SuperBowl",  3:"RetailStore", 4:"WebOffer"}

# Define a udf
udfIndexToCampaign = udf(lambda x: campaign_map[x], StringType())

result_df = (cust_df.select(udfIndexToCampaign("campaign").alias("campaign"))
                 .groupBy('campaign')
                 .count()
                 .orderBy('count', ascending=True))

result_df.cache().show()

result_df.toPandas().plot.barh(x='campaign');

result_df.unpersist()

result_df = (music_df.select(['customer_id', 'track_id'])
                            .groupBy('customer_id')
                            .agg(F.countDistinct('track_id').alias('unique_track_count'))
                            .join(cust_df, on='customer_id', how='inner')
                            .select([udfIndexTolevel('level').alias('level'), 'unique_track_count'])
                            .groupBy('level')
                            .agg(F.avg('unique_track_count').alias('avg_unique_track_count')))

result_df.cache().show()

result_df.toPandas().sort_values(by='avg_unique_track_count', ascending=False).plot.barh(x='level');

result_df.unpersist()

result_df = (music_df.select(['customer_id', 'track_id'])
                            .filter(col('is_mobile') == 1)
                            .groupBy('customer_id')
                            .count()
                            .withColumnRenamed('count', 'mobile_track_count')
                            .join(cust_df, on='customer_id', how='inner')
                            .select([udfIndexTolevel('level').alias('level'), 'mobile_track_count'])
                            .groupBy('level')
                            .agg(F.avg('mobile_track_count').alias('avg_mobile_track_count'))
                            .orderBy('avg_mobile_track_count'))

result_df.cache().show()

result_df.toPandas().sort_values(by='avg_mobile_track_count', ascending=False).plot.barh(x='level');

result_df.unpersist()

music_df.unpersist()
cust_df.unpersist()

spark.stop()



