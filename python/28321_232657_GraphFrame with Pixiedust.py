cloudantHost='dtaieb.cloudant.com'
cloudantUserName='weenesserliffircedinvers'
cloudantPassword='72a5c4f939a9e2578698029d2bb041d775d088b5'

airports = sqlContext.read.format("com.cloudant.spark").option("cloudant.host",cloudantHost)    .option("cloudant.username",cloudantUserName).option("cloudant.password",cloudantPassword)    .option("schemaSampleSize", "-1").load("flight-metadata")
airports.cache()
airports.registerTempTable("airports")

import pixiedust

# Display the airports data
display(airports)

flights = sqlContext.read.format("com.cloudant.spark").option("cloudant.host",cloudantHost)    .option("cloudant.username",cloudantUserName).option("cloudant.password",cloudantPassword)    .option("schemaSampleSize", "-1").load("pycon_flightpredict_training_set")
flights.cache()
flights.registerTempTable("training")

# Display the flights data
display(flights)

from pyspark.sql import functions as f
from pyspark.sql.types import *

rdd = flights.rdd.flatMap(lambda s: [s.arrivalAirportFsCode, s.departureAirportFsCode]).distinct()    .map(lambda row:[row])
vertices = airports.join(
      sqlContext.createDataFrame(rdd, StructType([StructField("fs",StringType())])), "fs"
    ).dropDuplicates(["fs"]).withColumnRenamed("fs","id")

print(vertices.count())

edges = flights.withColumnRenamed("arrivalAirportFsCode","dst")    .withColumnRenamed("departureAirportFsCode","src")    .drop("departureWeather").drop("arrivalWeather").drop("pt_type").drop("_id").drop("_rev")

print(edges.count())

import pixiedust

if sc.version.startswith('1.6.'):  # Spark 1.6
    pixiedust.installPackage("graphframes:graphframes:0.5.0-spark1.6-s_2.11")
elif sc.version.startswith('2.'):  # Spark 2.1, 2.0
    pixiedust.installPackage("graphframes:graphframes:0.5.0-spark2.1-s_2.11")


pixiedust.installPackage("com.typesafe.scala-logging:scala-logging-api_2.11:2.1.2")
pixiedust.installPackage("com.typesafe.scala-logging:scala-logging-slf4j_2.11:2.1.2")

print("done")

from graphframes import GraphFrame
g = GraphFrame(vertices, edges)
display(g)

from pyspark.sql.functions import *
degrees = g.degrees.sort(desc("degree"))
display( degrees )

r = g.shortestPaths(landmarks=["BOS", "LAX"]).select("id", "distances")
display(r)

from pyspark.sql.functions import *

ranks = g.pageRank(resetProbability=0.20, maxIter=5)

rankedVertices = ranks.vertices.select("id","pagerank").orderBy(desc("pagerank"))
rankedEdges = ranks.edges.select("src", "dst", "weight").orderBy(desc("weight") )

ranks = GraphFrame(rankedVertices, rankedEdges)
display(ranks)

paths = g.bfs(fromExpr="id='BOS'",toExpr="id = 'SFO'",edgeFilter="carrierFsCode='UA'", maxPathLength = 2)    .drop("from").drop("to")
paths.cache()
display(paths)

from pyspark.sql.functions import *

h = GraphFrame(g.vertices, g.edges.select("src","dst")   .groupBy("src","dst").agg(count("src").alias("count")))

query = h.find("(a)-[]->(b);(b)-[]->(c);!(a)-[]->(c)").drop("b")
query.cache()
display(query)

from pyspark.sql.functions import *
components = g.stronglyConnectedComponents(maxIter=10).select("id","component")    .groupBy("component").agg(count("id").alias("count")).orderBy(desc("count"))
display(components)

from pyspark.sql.functions import *
communities = g.labelPropagation(maxIter=5).select("id", "label")    .groupBy("label").agg(count("id").alias("count")).orderBy(desc("count"))
display(communities)

get_ipython().run_cell_magic('scala', '', 'import org.graphframes.lib.AggregateMessages\nimport org.apache.spark.sql.functions.{avg,desc,floor}\n\n// For each airport, average the delays of the departing flights\nval msgToSrc = AggregateMessages.edge("deltaDeparture")\nval __agg = g.aggregateMessages\n  .sendToSrc(msgToSrc)  // send each flight delay to source\n  .agg(floor(avg(AggregateMessages.msg)).as("averageDelays"))  // average up all delays\n  .orderBy(desc("averageDelays"))\n  .limit(10)\n__agg.cache()\n__agg.show()')

display(__agg)



