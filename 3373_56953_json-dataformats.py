import csv

airports = []

with open('/home/data_scientist/rppdm/data/airports.csv', 'r') as csvfile:
    
    for row in csv.reader(csvfile, delimiter=','):
        airports.append(row)

print(airports[0:3])

import json

with open('data.json', 'w') as fout:
    json.dump(airports, fout)

get_ipython().system('head data.json')

# First we can display the first few rows of the original data for comparison.

print(airports[:3], '\n', '-'*80)

# We use the pretty-print method to 
from pprint import pprint

# Open file and read the JSON formatted data
with open('data.json', 'r') as fin:
    data = json.load(fin)

# Pretty-print the first few rows
pprint(data[:3])

