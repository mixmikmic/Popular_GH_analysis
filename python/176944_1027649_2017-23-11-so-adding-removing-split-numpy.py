import numpy as np
# we can use the first argument [start]
arr = np.arange(5)
arr

# we can pass [start] stop[step]
arr2 = np.arange(1, 10)
print("array containing [start(1) - end(10)]: ",arr2)

#apply step
arr3 = np.arange(1, 10, 2)
print("array containing [start(1) - end(10) - step(2)]: ", arr3)

# we can print shape of the array and as well as dtype
shp = np.arange(1,10)
print("Shape of array: ",shp.shape )
# dtype
dty = np.arange(1,20)
print("Dtype: ", dty.shape)

# we can create 2-dimention array
d_2 = np.array([[1,2,3],[4,5,6]])
d_2
print("2D shape: ", d_2.shape)
# we can use random function
rnd = np.random.random(9).reshape(3,3)
rnd
print("random array: ", rnd.shape)

array_1 = np.array([[1, 2], [3, 4]])
array_2 = np.array([[5, 6], [7, 8]])
array_1

array_2 

# we can add array_2 as rows to the end of array_1
# axis 0 = rows
np.concatenate((array_1, array_2), axis=0)

# we can add array_2 as columns to end of array_1
# axis 1 = columns
np.concatenate((array_1, array_2), axis=1)

del_arry = np.array([[1,2,3],[4,5,6]])
del_arry

# column 2: [3 and 6]
# we can delete columm on index 2 of array
del_arry = np.delete(del_arry, 2, axis=1)
del_arry

# row 1: [4, 5, 6]
# we can delete row on index 1 of the array
del_arry = np.delete(del_arry, 1, axis=0)
del_arry

insert_array = np.array([[1,2,3],[4,5,6]])
# we can insert values into array index 6 - at the end
insert_array = np.insert(insert_array, 6, 10)
# we can also insert at the begining 
insert_array = np.insert(insert_array, 0, 100)
insert_array

# we can fill up the whole column given value
insert_2 = np.arange(0,9).reshape(3,3)
print("original array:")
print(insert_2)

# we can insert 0s in second column
insert_2 = np.insert(insert_2, 1, 0, axis=1)
print("\nafter inserting 0's on the first column:")
print(insert_2)

# we can also insert list as well
list_array = np.arange(0,9).reshape(3,3)
list_array = np.insert(list_array, [1], [[10],[10],[10]], axis=1)
list_array = np.insert(list_array, [1], 10, axis=0)
list_array

or_array = np.array([[1,2,3],[4,5,6]])
print("Orignal array:\n ",or_array)
#splits arr into 3 sub-arrays 
split_array = np.split(or_array, 2)
print("\nwe have our array splitted into two arrays")
split_array

copy_array = np.arange(16.0).reshape(4, 4)
#splits arr horizontally on the 5th index
print("copy array:\n",copy_array)

# we splits our array into horizontal on the given index
h_split = np.hsplit(copy_array, 2)
h_split

# we can also split array into vertical on the given index
h_split = np.vsplit(copy_array, 2)
h_split







