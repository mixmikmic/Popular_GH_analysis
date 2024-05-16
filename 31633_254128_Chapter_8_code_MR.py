get_ipython().system('hdfs dfs -mkdir -p /datasets')
get_ipython().system('wget -q http://www.gutenberg.org/cache/epub/100/pg100.txt     -O ../datasets/shakespeare_all.txt')
get_ipython().system('hdfs dfs -put -f ../datasets/shakespeare_all.txt /datasets/shakespeare_all.txt')
get_ipython().system('hdfs dfs -put -f ../datasets/hadoop_git_readme.txt /datasets/hadoop_git_readme.txt')
get_ipython().system('hdfs dfs -ls /datasets')

with open('mapper_hadoop.py', 'w') as fh:
    fh.write("""#!/usr/bin/env python

import sys

for line in sys.stdin:
    print "chars", len(line.rstrip('\\n'))
    print "words", len(line.split())
    print "lines", 1
    """)


with open('reducer_hadoop.py', 'w') as fh:
    fh.write("""#!/usr/bin/env python

import sys

counts = {"chars": 0, "words":0, "lines":0}

for line in sys.stdin:
    kv = line.rstrip().split()
    counts[kv[0]] += int(kv[1])

for k,v in counts.items():
    print k, v
    """) 

get_ipython().system('chmod a+x *_hadoop.py')

get_ipython().system('cat ../datasets/hadoop_git_readme.txt | ./mapper_hadoop.py | sort -k1,1 | ./reducer_hadoop.py')

get_ipython().system('hdfs dfs -mkdir -p /tmp')
get_ipython().system('hdfs dfs -rm -f -r /tmp/mr.out')

get_ipython().system('hadoop jar /usr/local/hadoop/share/hadoop/tools/lib/hadoop-streaming-2.6.4.jar -files mapper_hadoop.py,reducer_hadoop.py -mapper mapper_hadoop.py -reducer reducer_hadoop.py -input /datasets/hadoop_git_readme.txt -output /tmp/mr.out')


get_ipython().system('hdfs dfs -ls /tmp/mr.out')

get_ipython().system('hdfs dfs -cat /tmp/mr.out/part-00000')



with open("MrJob_job1.py", "w") as fh:
    fh.write("""
from mrjob.job import MRJob


class MRWordFrequencyCount(MRJob):

    def mapper(self, _, line):
        yield "chars", len(line)
        yield "words", len(line.split())
        yield "lines", 1

    def reducer(self, key, values):
        yield key, sum(values)


if __name__ == '__main__':
    MRWordFrequencyCount.run()    
    """)

get_ipython().system('python MrJob_job1.py ../datasets/hadoop_git_readme.txt')

get_ipython().system('python MrJob_job1.py -r hadoop hdfs:///datasets/hadoop_git_readme.txt')

with open("MrJob_job2.py", "w") as fh:
    fh.write("""
from mrjob.job import MRJob
from mrjob.step import MRStep
import re

WORD_RE = re.compile(r"[\w']+")


class MRMostUsedWord(MRJob):

    def steps(self):
        return [
            MRStep(mapper=self.mapper_get_words,
                   reducer=self.reducer_count_words),
            MRStep(mapper=self.mapper_word_count_one_key,
                   reducer=self.reducer_find_max_word)
        ]

    def mapper_get_words(self, _, line):
        # yield each word in the line
        for word in WORD_RE.findall(line):
            yield (word.lower(), 1)

    def reducer_count_words(self, word, counts):
        # send all (num_occurrences, word) pairs to the same reducer.
        yield (word, sum(counts))
    
    def mapper_word_count_one_key(self, word, counts):
        # send all the tuples to same reducer
        yield None, (counts, word)

    def reducer_find_max_word(self, _, count_word_pairs):
        # each item of word_count_pairs is a tuple (count, word),
        yield max(count_word_pairs)


if __name__ == '__main__':
    MRMostUsedWord.run()
""")

# This time is running on a big dataset
get_ipython().system('python MrJob_job2.py --quiet ../datasets/shakespeare_all.txt')

get_ipython().system('python MrJob_job2.py -r hadoop --quiet hdfs:///datasets/shakespeare_all.txt')



