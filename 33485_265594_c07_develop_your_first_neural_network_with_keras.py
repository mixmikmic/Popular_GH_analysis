from keras.models import Sequential
from keras.layers import Dense
import numpy as np

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

dataset = np.loadtxt("./data_set/pima-indians-diabetes.data", delimiter=',')
X = dataset[:, 0:8]
Y = dataset[:, 8]
print 'X', X.shape, X.dtype
print 'Y', Y.shape, Y.dtype

# create model
model = Sequential()
model.add(Dense(12, input_dim=8, init='uniform', activation='relu'))
model.add(Dense(8, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))


# compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

import time
start_time = time.time()
# fit model
model.fit(X, Y, nb_epoch=150, batch_size=10)
end_time = time.time()
print "Fit time Cost %s s"%(end_time - start_time)

# evaluate the model
scores = model.evaluate(X, Y)
print "Training Dataset %s: %.2f"%(model.metrics_names[0], scores[1])
print "Training Dataset %s: %.2f%%"%(model.metrics_names[1], scores[1]*100)

from theano import function, config, shared, sandbox  
import theano.tensor as T  
import numpy  
import time  
  
vlen = 10 * 30 * 768  # 10 x #cores x # threads per core  
iters = 1000  
  
rng = numpy.random.RandomState(22)  
x = shared(numpy.asarray(rng.rand(vlen), config.floatX))  
f = function([], T.exp(x))  
print(f.maker.fgraph.toposort())  
t0 = time.time()  
for i in xrange(iters):  
    r = f()  
t1 = time.time()  
print("Looping %d times took %f seconds" % (iters, t1 - t0))  
print("Result is %s" % (r,))  
if numpy.any([isinstance(x.op, T.Elemwise) for x in f.maker.fgraph.toposort()]):  
    print('Used the cpu')  
else:  
    print('Used the gpu')



