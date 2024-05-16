from sklearn import preprocessing
import numpy as np 

data = np.array([[2.2, 5.9, -1.8], [5.4, -3.2, -5.1], [-1.9, 4.2, 3.2]])

bindata = preprocessing.Binarizer(threshold=1.5).transform(data)
print('Binarized data:\n\n', bindata)





