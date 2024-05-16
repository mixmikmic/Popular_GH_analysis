# Create an list of items denoting the number of soldiers in each regiment, view the list
regimentSize = (5345, 6436, 3453, 2352, 5212, 6232, 2124, 3425, 1200, 1000, 1211); regimentSize

# Create a list called smallRegiments that filters regimentSize to 
# find all items that fulfill the lambda function (which looks for all items under 2500).
smallRegiments = list(filter((lambda x: x < 2500), regimentSize)); smallRegiments

# Create a lambda function that looks for things under 2500
lessThan2500Filter = lambda x: x < 2500

# Filter regimentSize by the lambda function filter
filteredRegiments = filter(lessThan2500Filter, regimentSize)

# Convert the filter results into a list
smallRegiments = list(filteredRegiments)

# Create a variable for the results of the loop to be placed
smallRegiments_2 = []

# for each item in regimentSize,
for x in regimentSize:
    # look if the item's value is less than 2500
    if x < 2500:
        # if true, add that item to smallRegiments_2
        smallRegiments_2.append(x)

# View the smallRegiment_2 variable
smallRegiments_2

