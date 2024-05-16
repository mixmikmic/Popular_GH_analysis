# Import combinations with replacements from itertools
from itertools import combinations_with_replacement

# Create a list of objects to combine
list_of_objects = ['warplanes', 'armor', 'infantry']

# Create an empty list object to hold the results of the loop
combinations = []

# Create a loop for every item in the length of list_of_objects, that,
for i in list(range(len(list_of_objects))):
    # Finds every combination (with replacement) for each object in the list
    combinations.append(list(combinations_with_replacement(list_of_objects, i+1)))
    
# View the results
combinations

# Flatten the list of lists into just a list
combinations = [i for row in combinations for i in row]

# View the results
combinations

