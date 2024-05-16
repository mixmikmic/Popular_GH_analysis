# Create a function that
def function(names):
    # For each name in a list of names
    for name in names:
        # Returns the name
        return name

# Create a variable of that function
students = function(['Abe', 'Bob', 'Christina', 'Derek', 'Eleanor'])

# Run the function
students

# Create a generator that
def generator(names):
    # For each name in a list of names
    for name in names:
        # Yields a generator object
        yield name

# Same as above, create a variable for the generator
students = generator(['Abe', 'Bob', 'Christina', 'Derek', 'Eleanor'])

# Run the generator
students

# Return the next student
next(students)

# Return the next student
next(students)

# Return the next student
next(students)

# List all remaining students in the generator
list(students)

