import nltk
import csv
import numpy as np

negative = []
with open("words_negative.csv", "rb") as file:
    reader = csv.reader(file)
    for row in reader:
        negative.append(row)

positive = []
with open("words_positive.csv", "rb") as file:
    reader = csv.reader(file)
    for row in reader:
        positive.append(row)

positive[:10]

negative[:10]

def sentiment(text):
    temp = [] #
    text_sent = nltk.sent_tokenize(text)
    for sentence in text_sent:
        n_count = 0
        p_count = 0
        sent_words = nltk.word_tokenize(sentence)
        for word in sent_words:
            for item in positive:
                if(word == item[0]):
                    p_count +=1
            for item in negative:
                if(word == item[0]):
                    n_count +=1

        if(p_count > 0 and n_count == 0): #any number of only positives (+) [case 1]
            #print "+ : " + sentence
            temp.append(1)
        elif(n_count%2 > 0): #odd number of negatives (-) [case2]
            #print "- : " + sentence
            temp.append(-1)
        elif(n_count%2 ==0 and n_count > 0): #even number of negatives (+) [case3]
            #print "+ : " + sentence
            temp.append(1)
        else:
            #print "? : " + sentence
            temp.append(0)
    return temp

sentiment("It was terribly bad.")

sentiment("Actualluty, it was not bad at all.")

sentiment("This is a sentance about nothing.")

mylist = sentiment("I saw this movie the other night. I can say I was not disappointed! The actiing and story line was amazing and kept me on the edge of my seat the entire time. While I did not care for the music, it did not take away from the overall experience. I would highly recommend this movie to anyone who enjoys thirllers.")

comments = []
with open("reviews.csv", "rb") as file:
    reader = csv.reader(file)
    for row in reader:
        comments.append(row)

comments[0]

for review in comments:
    print "\n"
    print np.average(sentiment(str(review)))
    print review

