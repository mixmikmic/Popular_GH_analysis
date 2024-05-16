import os
import json
import re
import time
from bs4 import BeautifulSoup 

# Should have 82.83 million reviews
filename = "raw/aggressive_dedup.json"
print(filename)
print(os.path.getsize(filename)/(1024*1024*1024), " GB")

def line_feeder(fname):
    with open(fname, 'r', encoding='utf-8') as f:
        for line in f:
            js_out = json.loads(line) 
            yield js_out   
        
def clean_review(review):
    temp = BeautifulSoup(review, "lxml").get_text()
    punctuation = "\"'.,?!:;(){}[]"
    for char in punctuation:
        temp = temp.replace(char, ' ' + char + ' ')
    words = " ".join(temp.lower().split()) + "\n"
    return words

def example(start=200, cut=5):
    for c,x in enumerate(line_feeder(filename)):
        if c > start:
            rev, rating = clean_review(x["reviewText"]), x["overall"]
            print("Raw:")
            print(x["reviewText"]), x["overall"]
            print("Clean:")
            print(rev, rating)          
            if c == start+cut:
                return
        
example()

good_rev = []
bad_rev = []
neut_rev = []
error_rev = []

gr = open('good_reviews.txt', 'w', encoding='utf-8')
br = open('bad_reviews.txt', 'w', encoding='utf-8')
nt = open('neutral_reviews.txt', 'w', encoding='utf-8')
er = open('error_reviews.txt', 'w', encoding='utf-8')

chunks = 0
stime = time.time()
for x in line_feeder(filename):
    
    chunks += 1
    rev, rating = clean_review(x["reviewText"]), x["overall"]
    
    if not len(rev) > 10:
        # Fewer than 10 characters not meangingful
        error_rev.append(rev)
    else:
        # Review long enough to consider
        if rating in [4,5]:
            good_rev.append(rev)
        elif rating in [1,2]:
            bad_rev.append(rev)
        else:
            neut_rev.append(rev)
            
    # Chunk every N=1000*000 reviews
    # Limited by IO, disk = 96%
    # Takes 305 seconds for 1mill, so around 420 minutes = 7 hours
    if chunks % (1000*1000) == 0:
        print("Processed: %d records" % chunks)
        print("Elapsed: %.2f" % (time.time() - stime))

        gr.writelines(good_rev)
        br.writelines(bad_rev)
        nt.writelines(neut_rev)
        er.writelines(error_rev)

        good_rev = []
        bad_rev = []
        neut_rev = []
        error_rev = []
            
# Any remaining
gr.writelines(good_rev)
gr.close()
br.writelines(bad_rev)
br.close()
nt.writelines(neut_rev)
nt.close()
er.writelines(error_rev)
er.close()

del good_rev
del bad_rev
del neut_rev
del error_rev

# Check sizes
def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

# Should add up:
# print("Raw contains %d lines" % file_len(filename))

# We have 64,439,865 good reviews
# print("Good contains %d lines" % file_len('good_reviews.txt'))
# We have 10,961,504 bad reviews
# print("Bad contains %d lines" % file_len('bad_reviews.txt'))


# print("Neutral contains %d lines" % file_len('neutral_reviews.txt'))
# print("Short contains %d lines" % file_len('error_reviews.txt'))

# 1 mill 
_SAMPLE_SIZE = 1000*1000

# Split data into train and test (also use subsample):
import random

def train_test_split(train_ratio=0.5):
    # Train -> true
    return random.uniform(0,1) <= train_ratio

def line_feeder(fname, cutoff):
    i = 0
    with open(fname, 'r', encoding='utf-8') as f:
        for line in f:
            yield line
            i+=1
            if i == cutoff:
                break
            
def split_data(dataname, sample_size, train_ratio):
    with open('train_' + dataname, 'w', encoding='utf-8') as tr:
        with open('test_' + dataname, 'w', encoding='utf-8') as te:
            for line in line_feeder(dataname, sample_size):
                if train_test_split(0.5):
                    tr.write(line)
                else:
                    te.write(line)

# I wanted a quick cut so I go by the first _SAMPLE_SIZE reviews, perhaps
# a better approach is to create a probability = _SAMPLE_SIZE/full_size
# and keep if random <= prob. and thus sample all lines
# however that would take a bit longer so have omitted.

split_data(dataname = 'good_reviews.txt', sample_size = _SAMPLE_SIZE, train_ratio = 0.5)
split_data(dataname = 'bad_reviews.txt', sample_size = _SAMPLE_SIZE, train_ratio = 0.5)

sources = {'test_bad_reviews.txt':'TE_B',
           'test_good_reviews.txt':'TE_G',
           'train_bad_reviews.txt':'TR_B',
           'train_good_reviews.txt':'TR_G'}

