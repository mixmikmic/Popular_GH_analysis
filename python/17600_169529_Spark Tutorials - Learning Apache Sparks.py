# Create RDD and subtract 1 from each number then find max
dataRDD = sc.parallelize(xrange(1,21))

# Let's see how many partitions the RDD will be split into using the getNumPartitions()
dataRDD.getNumPartitions()

dataRDD.map(lambda x: x - 1).max()
dataRDD.toDebugString()

# Find the even numbers
print(dataRDD.getNumPartitions())

# Find even numbers
evenRDD = dataRDD.filter(lambda x: x % 2 == 0)

# Reduce by adding up all values in the RDD
print(evenRDD.reduce(lambda x, y: x + y))


# Use Python add function to sum
from operator import add
print(evenRDD.reduce(add))



# Take first n values
evenRDD.take(10)

# Count distinct values in RDD and return dictionary of values and counts
evenRDD.countByValue()

pairRDD = sc.parallelize([('a', 1), ('a', 2), ('b', 1)])

# mapValues only used to improve format for printing
print pairRDD.groupByKey().mapValues(lambda x: list(x)).collect()

# Different ways to sum by key
print pairRDD.groupByKey().map(lambda (k, v): (k, sum(v))).collect()

# Using mapValues, which is recommended when the key doesn't change
print pairRDD.groupByKey().mapValues(lambda x: sum(x)).collect()

# reduceByKey is more efficient / scalable
print pairRDD.reduceByKey(add).collect()

# mapPartitions takes a function that takes an iterator and returns an iterator
print wordsRDD.collect()

itemsRDD = wordsRDD.mapPartitions(lambda iterator: [','.join(iterator)])

print itemsRDD.collect()

itemsByPartRDD = wordsRDD.mapPartitionsWithIndex(lambda index, iterator: [(index, list(iterator))])

# We can see that three of the (partitions) workers have one element and the fourth worker has two
# elements, although things may not bode well for the rat...
print itemsByPartRDD.collect()

# Rerun without returning a list (acts more like flatMap)
itemsByPartRDD = wordsRDD.mapPartitionsWithIndex(lambda index, iterator: (index, list(iterator)))

print itemsByPartRDD.collect()

def brokenTen(value):
    """Incorrect implementation of the ten function.

    Note:
        The `if` statement checks an undefined variable `val` instead of `value`.

    Args:
        value (int): A number.

    Returns:
        bool: Whether `value` is less than ten.

    Raises:
        NameError: The function references `val`, which is not available in the local or global
            namespace, so a `NameError` is raised.
    """
#     if (val < 10):
    if (value < 10):
        return True
    else:
        return False

brokenRDD = dataRDD.filter(brokenTen)

brokenRDD.collect()

wordslist = ['cat', 'elephant', 'rat', 'rat', 'cat']
wordsRDD = sc.parallelize(wordslist)
print(type(wordsRDD))

# Pluralize each word
pluralwords = wordsRDD.map(lambda w: w +'s').collect()
pluralwords

# Find length of each word
pluralRDD = sc.parallelize(pluralwords)
pluralRDD.map(len).collect()

# CountByValue()
pluralRDD.countByValue().items()

# Lambda with CountByKey()
pluralRDD.map(lambda d: (d,1)).countByKey().items()

# Count with Lambda with reduceByKey()
newPluralRDD = pluralRDD.map(lambda d: (d,1))
print(newPluralRDD.collect())

newPluralRDD.reduceByKey(add).collect()

print(pluralRDD.collect())
pluralRDD.distinct().collect()

total = pluralRDD.count()
size = float(pluralRDD.distinct().count())
round(total/size, 2)


def wordCount(wordsListRDD):
    """ Inputs word List RDD and outputs Key value pair count of the words."""
    return wordsListRDD.countByValue().items()

wordCount(wordsRDD)


import re
f = open("alice.txt", 'r')

pattern = re.compile(r"[.,\[\]\"'*!?`_\s();-]+")
wordsList = [re.sub(pattern, '', sents).lower() for sents in f.read().split()]
wordsList = filter(None, wordsList)


wordsFileRDD = sc.parallelize(wordsList)
p_wordfile = wordCount(wordsFileRDD)

# Print top 30 words
sorted(p_wordfile, key=lambda tup:tup[1], reverse=True)[:30]



