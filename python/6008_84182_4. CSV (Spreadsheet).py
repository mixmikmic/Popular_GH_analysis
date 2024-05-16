import nltk
import csv

comments = []
with open("reviews.csv", "rb") as file:
    reader = csv.reader(file)
    for row in reader:
        comments.append(row)

comments[0]

tokens = [nltk.word_tokenize(str(entry)) for entry in comments]

tokens[0]

