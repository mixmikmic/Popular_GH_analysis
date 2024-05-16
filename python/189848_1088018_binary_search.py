sorted_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20]
print(sorted_list)

def binary_search(sorted_list, target):

    '''This function inputs a sorted list and a target value to find and returns ....'''

    # Create variables for the index of the first and last elements
    start = 0
    end = len(sorted_list) - 1

    while end >= start:

        # Create a variable for the index of the middle term
        middle = start + (end - start) // 2  # // is integer division in Python 3.X

        # If the target value is less than the middle value of the search area
        if target < sorted_list[middle]:
            # Cut the list in half by making the new end value the old middle value minus 1
            # The minus one is because we already ruled the middle value out, so we can ignore it
            end = middle - 1

        #  Else, if the target value is greater than the middle value of the search area
        elif target > sorted_list[middle]:
            # Cut the list in half by making the new start value the old middle value plus 1
            # The plus one is because we already ruled the middle value out, so we can ignore it
            start = middle + 1

        # If it's not too high or too low, it must be just right, return the location
        else:
            return ("Found it at index: {}".format(middle))

    # If we've fallen out of the while loop the target value is not in the list
    return print("Not in list!")

# Run binary search
binary_search(sorted_list, 2)

