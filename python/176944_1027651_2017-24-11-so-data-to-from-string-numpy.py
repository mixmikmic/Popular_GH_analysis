import numpy as np

a = np.array([[1,2],
           [3,4]], 
          dtype = np.uint8)

a.tostring()

a.tostring(order='F')

s = a.tostring()
a = np.fromstring(s, dtype=np.uint8)
a

a.shape = 2,2
a



