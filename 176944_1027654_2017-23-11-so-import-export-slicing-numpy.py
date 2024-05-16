import numpy as np 
# using numpy you can load text file
np.loadtxt('file_name.txt')
# load csv file
np.genfromtxt('file_name.csv', delimiter=',')
# you can write to a text file and save it
np.savetxt('file_name.txt', arr, delimiter=' ')
# you can write to a csv file and save it
np.savetxt('file_name.csv', arr, delimiter=',')

# slicing 1 to 7 gives us: [1 through 6]
slice_array = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
slice_array[1:7]

# if we do this, we are giving k, which is the step function. in this case step by 2
slice_array[1:7:2]

slice_arrays = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])+1
#return the element at index 5
slice_arrays[5]

#returns the 2D array element on index value of 2 and 5
slice_arrays[[2,5]]

#assign array element on index 1 the value 4
slice_arrays[1] = 100
#assign array element on index [1][3] the value 10
slice_arrays[[1,3]] = 100
slice_arrays

#return the elements at indices 0,1,2 on a 2D array:
slice_arrays[0:3]

#returns the elements at indices 1,100
slice_arrays[:2]

slice_2d = np.arange(16).reshape(4,4)
slice_2d

#returns the elements on rows 0,1,2, at column 4
slice_2d[0:3, :4]

#returns the elements at index 1 on all columns
slice_2d[:, 1]

# return the last two rows
slice_2d[-2:10]
# returns the last three rows
slice_2d[1:]

# reverse all the array backword
slice_2d[::-1]

#returns an array with boolean values
slice_2d < 5

#inverts a boolearn array, if its positive arr - convert to negative, vice versa
~slice_2d

#returns array elements smaller than 5
slice_2d[slice_2d < 5]



