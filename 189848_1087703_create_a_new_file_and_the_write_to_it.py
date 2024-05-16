# Create a file if it doesn't already exist
with open('file.txt', 'xt') as f:
    # Write to the file
    f.write('This file now exsits!')
    # Close the connection to the file
    f.close()

# Open the file
with open('file.txt', 'rt') as f:
    # Read the data in the file
    data = f.read()
    # Close the connection to the file
    f.close()

# View the data in the file
data

# Import the os package
import os

# Delete the file
os.remove('file.txt')

