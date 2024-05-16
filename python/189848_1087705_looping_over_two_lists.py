# Create a list of length 3:
armies = ['Red Army', 'Blue Army', 'Green Army']

# Create a list of length 4:
units = ['Red Infantry', 'Blue Armor','Green Artillery','Orange Aircraft']

# For each element in the first list,
for army, unit in zip(armies, units):
    # Display the corresponding index element of the second list:
    print(army, 'has the following options:', unit)

