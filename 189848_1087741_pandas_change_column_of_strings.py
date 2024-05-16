# Import pandas
import pandas as pd

# Create a list of first names
first_names = pd.Series(['Steve Murrey', 'Jane Fonda', 'Sara McGully', 'Mary Jane'])

# print the column
first_names

# print the column with lower case
first_names.str.lower()

# print the column with upper case
first_names.str.upper()

# print the column with title case
first_names.str.title()

# print the column split across spaces
first_names.str.split(" ")

# print the column with capitalized case
first_names.str.capitalize()

