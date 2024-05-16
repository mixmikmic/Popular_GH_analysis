# Create a dictionary of arguments
argument_dict = {'a':'Alpha', 'b':'Bravo'}

# Create a list of arguments
argument_list = ['Alpha', 'Bravo']

# Create a function that takes two inputs
def simple_function(a, b):
    # and prints them combined
    return a + b

# Run the function with the unpacked argument dictionary
simple_function(**argument_dict)

# Run the function with the unpacked argument list
simple_function(*argument_list)

