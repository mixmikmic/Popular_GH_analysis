import os
import pandas as pd
import numpy as np

from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession, SQLContext

from pyspark.sql.types import *
from pyspark.sql.window import Window

import pyspark.sql.functions as F
from pyspark.sql.functions import udf, col

pd.set_option('display.max_columns', 20)
pd.set_option('display.max_colwidth', 400)

# setting random seed for notebook reproducability
rnd_seed=23
np.random.seed=rnd_seed
np.random.set_state=rnd_seed

spark = SparkSession.builder.master("local[*]").appName("nested_attributes").getOrCreate()

spark

sc = spark.sparkContext
sc

sqlContext = SQLContext(spark.sparkContext)
sqlContext

import re

# Utility function to emulate stripMargin in Scala string.
def strip_margin(text):
    nomargin = re.sub('\n[ \t]*\|', ' ', text)
    trimmed = re.sub('\s+', ' ', nomargin)
    return trimmed

spotify_df = spark.read.csv(path='data/spotify-songs.csv', inferSchema=True, header=True).cache()

spotify_df.limit(10).toPandas()

map_df = (spotify_df
          .select('id', 'song_title', 'artist', 'duration_ms',
                  F.array('key', 'mode', 'target').alias('audience'), 
                  F.create_map(
                      F.lit('acousticness'), 'acousticness', 
                      F.lit('danceability'), 'acousticness',
                      F.lit('energy'), 'energy',
                      F.lit('instrumentalness'), 'instrumentalness',
                      F.lit('liveness'), 'liveness',
                      F.lit('loudness'), 'loudness',
                      F.lit('speechiness'), 'speechiness',
                      F.lit('tempo'), 'tempo'
                  ).alias('qualities'),
                 'time_signature',
                 'valence')
        .cache())

map_df.limit(10).toPandas()

# Let's check the schema of the new DataFrame
map_df.printSchema()

map_df.write.json(path='data/spotify-songs', mode="overwrite")

nested_schema = StructType([
    StructField('id', IntegerType(), nullable=False),
    StructField('song_title', StringType(), nullable=False),
    StructField('artist', StringType(), nullable=False),
    StructField('duration_ms', IntegerType(), nullable=False),
    StructField('audience', ArrayType(elementType=IntegerType()), nullable=False),
    StructField('qualities', MapType(keyType=StringType(), valueType=DoubleType(), valueContainsNull=False), nullable=True),
    StructField('time_signature', IntegerType(), nullable=False),
    StructField('valence', DoubleType(), nullable=False),
  ])

spotify_reloaded_df = spark.read.json(path='data/spotify-songs', schema=nested_schema).cache()

spotify_reloaded_df.limit(10).toPandas()

spotify_reloaded_df.printSchema()

(spotify_reloaded_df
 .select(spotify_reloaded_df.audience.getItem(0).alias('key'), 
         spotify_reloaded_df.audience.getItem(1).alias('mode'),
         spotify_reloaded_df.audience.getItem(2).alias('target'))
 .limit(10)
 .toPandas())

(spotify_reloaded_df
 .select(
     spotify_reloaded_df.qualities.getItem('acousticness').alias('acousticness'),
     spotify_reloaded_df.qualities.getItem('speechiness').alias('speechiness')
 )
 .limit(10)
 .toPandas())

(spotify_reloaded_df
 .select('id', 'song_title', 'artist',
     spotify_reloaded_df.qualities.getItem('acousticness').alias('acousticness'),
     spotify_reloaded_df.qualities.getItem('danceability').alias('danceability'),
     'duration_ms',
     spotify_reloaded_df.qualities.getItem('energy').alias('energy'),
     spotify_reloaded_df.qualities.getItem('instrumentalness').alias('instrumentalness'),
     spotify_reloaded_df.audience.getItem(0).alias('key'),
     spotify_reloaded_df.qualities.getItem('liveness').alias('liveness'),
     spotify_reloaded_df.qualities.getItem('loudness').alias('loudness'),
     spotify_reloaded_df.audience.getItem(1).alias('mode'),
     spotify_reloaded_df.qualities.getItem('speechiness').alias('speechiness'),
     spotify_reloaded_df.qualities.getItem('tempo').alias('tempo'),
     'time_signature',
     'valence',
     spotify_reloaded_df.audience.getItem(2).alias('target')
 )
 .limit(10)
 .toPandas())

(spotify_reloaded_df
 .select(F.posexplode(spotify_reloaded_df.audience))
 .limit(10)
 .toPandas())

(spotify_reloaded_df
 .select(F.explode(spotify_reloaded_df.qualities).alias("qualities", "value"))
 .limit(10)
 .toPandas())

spark.stop()

