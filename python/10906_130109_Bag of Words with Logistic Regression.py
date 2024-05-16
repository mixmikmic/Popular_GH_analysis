

import pandas as pd
from bs4 import BeautifulSoup
import nltk
import numpy as np
from sklearn import linear_model

data = pd.read_csv("labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)

#do a simple split by using "ID" with trailing 3 before the _

train = data[data["id"].str.contains("3_")==False]
test = data[data["id"].str.contains("3_")==True]

print train.shape, test.shape

2500/25000.0 *100 #so split is perfect 10%

train.head()

test.head()

sum(train["sentiment"])/22500.0 #training set is nicely half positive and half negative as well

#The tutorial goes through the steps in the function to show what each is doing they are...

# Initialize the BeautifulSoup object on a single movie review     
example1 = BeautifulSoup(train["review"][0])  

# Print the raw review and then the output of get_text(), for 
# comparison
print train["review"][0]
print example1.get_text()

import re
# Use regular expressions to do a find-and-replace
letters_only = re.sub("[^a-zA-Z]",           # The pattern to search for
                      " ",                   # The pattern to replace it with
                      example1.get_text() )  # The text to search
print letters_only

lower_case = letters_only.lower()        # Convert to lower case
words = lower_case.split()               # Split into words

#Some of the most common stopwords, you would normally get this through a package
stopwords  = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here',
'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now']

#The Main function is review_to_words which does all the text process of cleaning and splitting

def review_to_words( raw_review ):
    # Function to convert a raw review to a string of words
    # The input is a single string (a raw movie review), and 
    # the output is a single string (a preprocessed movie review)
    #
    # 1. Remove HTML
    review_text = BeautifulSoup(raw_review).get_text() 
    #
    # 2. Remove non-letters        
    letters_only = re.sub("[^a-zA-Z]", " ", review_text) 
    #
    # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split()                             
    #
    # 4. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    stops = stopwords                  
    # 
    # 5. Remove stop words
    meaningful_words = [w for w in words if not w in stops]   
    #
    # 6. Join the words back into one string separated by space, 
    # and return the result.
    return( " ".join( meaningful_words ))   

clean_review = review_to_words( train["review"][0] )
print clean_review

train_labels = train["sentiment"]
test_labels = test["sentiment"]

print "Cleaning and parsing the training set movie reviews...\n"
# Get the number of reviews based on the dataframe column size
num_reviews = data["review"].size

# Initialize an empty list to hold the clean reviews
#When the data was split the train, test sets kept the index which we will use to our advantage here
clean_train_reviews = []
clean_test_reviews = []
for i in xrange( 0, num_reviews ):
    # If the index is evenly divisible by 1000, print a message
    if( (i+1)%1000 == 0 ):
        print "Review %d of %d\n" % ( i+1, num_reviews )
    try:
        clean_train_reviews.append(review_to_words(train["review"][i] ))
    except:
        clean_test_reviews.append(review_to_words(test["review"][i] ))

print "Creating the bag of words...\n"
from sklearn.feature_extraction.text import CountVectorizer

# Initialize the "CountVectorizer" object, which is scikit-learn's
# bag of words tool.  
vectorizer = CountVectorizer(analyzer = "word",                                tokenizer = None,                                 preprocessor = None,                              stop_words = None,                                max_features = 5000) 

# fit_transform() does two functions: First, it fits the model
# and learns the vocabulary; second, it transforms our training data
# into feature vectors. The input to fit_transform should be a list of 
# strings.
train_data_features = vectorizer.fit_transform(clean_train_reviews)

# Numpy arrays are easy to work with, so convert the result to an 
# array
train_data_features = train_data_features.toarray()

#Also make the test data into the correct format
test_data_features = vectorizer.fit_transform(clean_test_reviews)

# Numpy arrays are easy to work with, so convert the result to an 
# array
test_data_features = test_data_features.toarray()

print train_data_features.shape

print sum(train_data_features[0]), max(train_data_features[0]), train_data_features[0]

print sum(test_data_features[0]), max(test_data_features[0]), test_data_features[0]

#try using the BagOfWords with the logistic regression
logreg = linear_model.LogisticRegression(C=1e5)

# we create an instance of Neighbours Classifier and fit the data.
logreg.fit(train_data_features, train_labels)

#Now that we have something trained we can check if it is accurate with the test set

preds = logreg.predict(test_data_features)

#because the label is zero or one the root difference is simply the absolute difference between predicted and actual
rmse = sum(abs(preds-test_labels))/float(len(test_labels)) 
print rmse

#Not a very good model as it is just every so slightly better than random



#Some additional data analysis of the vocabulary and model...

# Take a look at the words in the vocabulary
vocab = vectorizer.get_feature_names()
print vocab[:10]

import numpy as np

# Sum up the counts of each vocabulary word
dist = np.sum(train_data_features, axis=0)

# For each, print the vocabulary word and the number of times it 
# appears in the training set
for tag, count in zip(vocab, dist):
    print count, tag

