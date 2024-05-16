import pixiedust
pixiedust.printAllPackages()

pixiedust.installPackage("graphframes:graphframes:0")

pixiedust.printAllPackages()

#import the Graphs example
from graphframes.examples import Graphs
#create the friends example graph
g=Graphs(sqlContext).friends()
#use the pixiedust display
display(g)

pixiedust.installPackage("org.apache.commons:commons-csv:0")

pixiedust.installPackage("https://github.com/ibm-watson-data-lab/spark.samples/raw/master/dist/streaming-twitter-assembly-1.6.jar")

pixiedust.uninstallPackage("org.apache.commons:commons-csv:0")

