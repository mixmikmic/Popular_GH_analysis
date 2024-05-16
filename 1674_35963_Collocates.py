get_ipython().magic('ls *.txt')

targetText = "Hume Treatise.txt"

with open(targetText, "r") as f:
    theText = f.read()

print("This string has", "{:,}".format(len(theText)), "characters")

import re
theTokens = re.findall(r'\b\w[\w-]*\b', theText.lower())
print(theTokens[:10])

wrd2find = input("What word do you want collocates for?") # Ask for the word to search for
context = 5 # This sets the context of words on either side to grab

end = len(theTokens)
counter = 0
theCollocates = []
for word in theTokens:
    if word == wrd2find: # This checks to see if the word is what we want
        for i in range(context):
            if (counter - (i + 1)) >= 0: # This checks that we aren't at the beginning
                theCollocates.append(theTokens[(counter - (i + 1))]) # This adds words before
            if (counter + (i + 1)) < end: # This checks that we aren't at the end
                theCollocates.append(theTokens[(counter + (i + 1))]) # This adds words afte
    counter = counter + 1
    
print(theCollocates[:10])

print(len(theCollocates))

print(set(theCollocates))

import nltk
tokenDist = nltk.FreqDist(theCollocates)
tokenDist.tabulate(10)

import matplotlib
get_ipython().magic('matplotlib inline')
tokenDist.plot(25, title="Top Frequency Collocates for " + wrd2find.capitalize())

import csv
nameOfResults = wrd2find.capitalize() + ".Collocates.csv"
table = tokenDist.most_common()

with open(nameOfResults, "w") as f:
    writer = csv.writer(f)
    writer.writerows(table)
    
print("Done")



