#!pip install --user --upgrade pixiedust

import pixiedust

pixiedust.enableJobMonitor();

pixiedust.installPackage("graphframes:graphframes:0.1.0-spark1.6")
print("done")

pixiedust.printAllPackages()

sqlContext=SQLContext(sc)
d1 = sqlContext.createDataFrame(
[(2010, 'Camping Equipment', 3),
 (2010, 'Golf Equipment', 1),
 (2010, 'Mountaineering Equipment', 1),
 (2010, 'Outdoor Protection', 2),
 (2010, 'Personal Accessories', 2),
 (2011, 'Camping Equipment', 4),
 (2011, 'Golf Equipment', 5),
 (2011, 'Mountaineering Equipment',2),
 (2011, 'Outdoor Protection', 4),
 (2011, 'Personal Accessories', 2),
 (2012, 'Camping Equipment', 5),
 (2012, 'Golf Equipment', 5),
 (2012, 'Mountaineering Equipment', 3),
 (2012, 'Outdoor Protection', 5),
 (2012, 'Personal Accessories', 3),
 (2013, 'Camping Equipment', 8),
 (2013, 'Golf Equipment', 5),
 (2013, 'Mountaineering Equipment', 3),
 (2013, 'Outdoor Protection', 8),
 (2013, 'Personal Accessories', 4)],
["year","zone","unique_customers"])

display(d1)

python_var = "Hello From Python"
python_num = 10

get_ipython().run_cell_magic('scala', '', 'println(python_var)\nprintln(python_num+10)\nval __scala_var = "Hello From Scala"')

print(__scala_var)

pixiedust.sampleData()

pixiedust.installPackage("com.databricks:spark-csv_2.10:1.5.0")
pixiedust.installPackage("org.apache.commons:commons-csv:0")

d2 = pixiedust.sampleData(1)

display(d2)

d3 = pixiedust.sampleData("https://openobjectstore.mybluemix.net/misc/milliondollarhomes.csv")

get_ipython().magic('pixiedustLog -l debug')

get_ipython().run_cell_magic('scala', '', 'val __scala_version = util.Properties.versionNumberString')

import platform
print('PYTHON VERSON = ' + platform.python_version())
print('SPARK VERSON = ' + sc.version)
print('SCALA VERSON = ' + __scala_version)

