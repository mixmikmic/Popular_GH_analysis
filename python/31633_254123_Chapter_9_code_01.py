# Example: let's encode the gender found in the demographic data
# As a hot encode. Note: the association should be the same
# on every machine in the cluster, requiring a shared mapping

one_hot_encoding = {"M": (1, 0, 0),
                    "F": (0, 1, 0),
                    "U": (0, 0, 1)
                   }

# Gender one-hot-encoding
(sc.parallelize(["M", "F", "U", "F", "M", "U"])
   .map(lambda x: one_hot_encoding[x])
   .collect())

# The command above works only in the single node configuration
# since the variable "one_hot_encoding" is defined only on this machine
# On a multi-node cluster, it will raise a Java error

# Solution 1: include the encoding map in the .map() function 
# In this way, all the nodes will see it

def map_ohe(x):
    ohe = {"M": (1, 0, 0),
           "F": (0, 1, 0),
           "U": (0, 0, 1)
          }
    return ohe[x]

sc.parallelize(["M", "F", "U", "F", "M", "U"]).map(map_ohe).collect()


# Solution 2: broadcast the map to all the nodes.
# All of them will be able to read-only it

bcast_map = sc.broadcast(one_hot_encoding)

def bcast_map_ohe(x, shared_ohe):
    return shared_ohe[x]

(sc.parallelize(["M", "F", "U", "F", "M", "U"])
 .map(lambda x: bcast_map_ohe(x, bcast_map.value))
 .collect())

bcast_map.unpersist()

# Let's coint the empty line in a file

print "The number of empty lines is:"

(sc.textFile('file:///home/vagrant/datasets/hadoop_git_readme.txt')
   .filter(lambda line: len(line) == 0)
   .count())

# Let's count the lines in a file, and at the same time,
# count the empty ones

accum = sc.accumulator(0)

def split_line(line):   
    if len(line) == 0:
        accum.add(1)
    return 1

tot_lines = (
    sc.textFile('file:///home/vagrant/datasets/hadoop_git_readme.txt')
      .map(split_line)
      .count())

empty_lines = accum.value


print "In the file there are %d lines" % tot_lines
print "And %d lines are empty" % empty_lines

# step 1: load the dataset
# note: if the dataset is large, you should read the next section

from sklearn.datasets import load_iris

bcast_dataset = sc.broadcast(load_iris())

# step 2: create an accumulator that stores the errors in a list

from pyspark import AccumulatorParam

class ErrorAccumulator(AccumulatorParam):
    def zero(self, initialList):
        return initialList

    def addInPlace(self, v1, v2):
        if not isinstance(v1, list):
            v1 = [v1]
        if not isinstance(v2, list):
            v2 = [v2]
        return v1 + v2

errAccum = sc.accumulator([], ErrorAccumulator())

# step 3: create mappers: each of them will use a classifier

def apply_classifier(clf, dataset):
    
    clf_name = clf.__class__.__name__
    X = dataset.value.data
    y = dataset.value.target
    
    try:
        from sklearn.metrics import accuracy_score
        
        clf.fit(X, y)
        y_pred = clf.predict(X)
        acc = accuracy_score(y, y_pred)

        return [(clf_name, acc)]

    except Exception as e:
        errAccum.add((clf_name, str(e)))
        return []

from sklearn.linear_model import SGDClassifier
from sklearn.dummy import DummyClassifier
from sklearn.decomposition import PCA
from sklearn.manifold import MDS

classifiers = [DummyClassifier('most_frequent'), 
               SGDClassifier(), 
               PCA(), 
               MDS()]

(sc.parallelize(classifiers)
     .flatMap(lambda x: apply_classifier(x, bcast_dataset))
     .collect())

print "The errors are:"
errAccum.value

bcast_dataset.unpersist()

from pyspark.sql import SQLContext
sqlContext = SQLContext(sc)

get_ipython().system('cat /home/vagrant/datasets/users.json')

df = sqlContext.read.json("file:///home/vagrant/datasets/users.json")
df.show()

df.printSchema()

(df.filter(df['gender'] != 'null')
   .filter(df['balance'] > 0)
   .select(['balance', 'gender', 'user_id'])
   .show())

(df.filter('gender is not null')
   .filter('balance > 0').select("*").show())

df.filter('gender is not null and balance > 0').show()

df.na.drop().show()

df.na.drop(subset=["gender"]).show()

df.na.fill({'gender': "U", 'balance': 0.0}).show()

(df.na.fill({'gender': "U", 'balance': 0.0})
   .groupBy("gender").avg('balance').show())

df.registerTempTable("users")

sqlContext.sql("""
    SELECT gender, AVG(balance) 
    FROM users 
    WHERE gender IS NOT NULL 
    GROUP BY gender""").show()

type(sqlContext.table("users"))

sqlContext.table("users").collect()

a_row = sqlContext.sql("SELECT * FROM users").first()
a_row

print a_row['balance']
print a_row.balance

a_row.asDict()

get_ipython().system('rm -rf /tmp/complete_users*')

(df.na.drop().write
   .save("file:///tmp/complete_users.json", format='json'))

get_ipython().system('ls -als /tmp/complete_users.json')

sqlContext.sql(
    "SELECT * FROM json.`file:///tmp/complete_users.json`").show()

df.na.drop().write.save(
    "file:///tmp/complete_users.parquet", format='parquet')

get_ipython().system('ls -als /tmp/complete_users.parquet/')

from pyspark.sql import Row

rdd_gender =     sc.parallelize([Row(short_gender="M", long_gender="Male"),
                    Row(short_gender="F", long_gender="Female")])

(sqlContext.createDataFrame(rdd_gender)
           .registerTempTable("gender_maps"))

sqlContext.table("gender_maps").show()

sqlContext.sql("""
    SELECT balance, long_gender, user_id 
    FROM parquet.`file:///tmp/complete_users.parquet` 
    JOIN gender_maps ON gender=short_gender""").show()

sqlContext.tableNames()

for table in sqlContext.tableNames():
    sqlContext.dropTempTable(table)



