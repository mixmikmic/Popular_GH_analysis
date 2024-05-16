# Make sure you have the latest version of PixieDust installed on your system
# Only run this cell if you did _not_ install PixieDust from source
# To confirm you have the latest, uncomment the next line and run this cell
#!pip install --user --upgrade pixiedust

# Run this cell
import pixiedust

# Run this cell to
# a) build a SQL context for a Spark dataframe 
sqlContext=SQLContext(sc) 
# b) create Spark dataframe, and assign it to a variable
df = sqlContext.createDataFrame(
[("Green", 75),
 ("Blue", 25)],
["Colors","%"])

# Run this cell to display the dataframe above as a pie chart
display(df)

# create another dataframe, in a new variable
df2 = sqlContext.createDataFrame(
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
["year","category","unique_customers"])

# This time, we've combined the dataframe and display() call in the same cell
# Run this cell 
display(df2)

# load a CSV with pixiedust.sampledata()
df3 = pixiedust.sampleData("https://github.com/ibm-watson-data-lab/open-data/raw/master/cars/cars.csv")
display(df3)

# To install Seaborn, uncomment the next line, and then run this cell
#!pip install --user seaborn

