get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import numpy as np

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, RMSprop, Adadelta, Adagrad, Adam

from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn import cross_validation
from sklearn.decomposition import PCA


iris = load_iris()

X_org, y = iris.data, iris.target
print "Classes present in IRIS", iris.target_names

# Convert y to one hot vector for each category
enc = OneHotEncoder()
y= enc.fit_transform(y[:, np.newaxis]).toarray()

# **VERY IMPORTANT STEP** Scale the values so that mean is 0 and variance is 1.
# If this step is not performed the Neural network will not converge. The logistic regression model might converge.
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

X.shape, y.shape

# Implement Model in Keras
model = Sequential()
model.add(Dense(X.shape[1], 2, init='uniform', activation='tanh'))
model.add(Dropout(0.5))
model.add(Dense(2, 2, init='uniform', activation='tanh'))
model.add(Dropout(0.5))
model.add(Dense(2, y.shape[1], init='uniform', activation='softmax'))

#sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
sgd = SGD()

# Compile the model using theano
model.compile(loss='categorical_crossentropy', optimizer="rmsprop")

"""
Use this cell only if you want to reset the weights of the model. 
"""
"""
#print model.get_weights()
model.set_weights(np.array([np.random.uniform(size=k.shape) for k in model.get_weights()]))
print model.to_yaml()
model.optimizer.lr = 0.01
model.optimizer.decay = 0.
model.optimizer.momentum = 0.
model.optimizer.nesterov = False
"""
# Done

# Perform cross validated training
cond = (y[:,0] == 1) | (y[:,1] == 1) | (y[:,2] == 1)
kf = cross_validation.KFold(X[cond].shape[0], n_folds=10, shuffle=True)
scores = []
for train_index, test_index in kf:
    model.fit(X[cond][train_index], y[cond][train_index], nb_epoch=10, batch_size=200, verbose=0)
    scores.append(model.evaluate(X[cond][test_index], y[cond][test_index], show_accuracy=1))
model.fit(X, y, nb_epoch=100, batch_size=200, verbose=0)
scores.append(model.evaluate(X, y, show_accuracy=1))
print scores
print np.mean(np.array(scores), axis=0)

print model.predict_classes(X[cond][test_index]), np.argmax(y[cond][test_index], axis=1)
print set(model.predict_classes(X))

logit = Sequential()
logit.add(Dense(X.shape[1], y.shape[1], init='uniform', activation='softmax'))
logit_sgd = SGD()

logit.compile(loss='categorical_crossentropy', optimizer=logit_sgd)

scores = []
kf = cross_validation.KFold(X.shape[0], n_folds=10)
for train_index, test_index in kf:
    logit.fit(X[train_index], y[train_index], nb_epoch=100, batch_size=200, verbose=0)
    scores.append(logit.evaluate(X[test_index], y[test_index], show_accuracy=1))
print scores
print np.mean(np.array(scores), axis=0)

pca = PCA(n_components=2)
X_t = pca.fit_transform(X)

h = 0.1
x_min, x_max = X_t[:, 0].min() - 1, X_t[:, 0].max() + 1
y_min, y_max = X_t[:, 1].min() - 1, X_t[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

fig, ax = plt.subplots(1,2, figsize=(20,10))
for i, v in enumerate({"Neural Net": model, "Logistic": logit}.items()):
    # here "model" is your model's prediction (classification) function
    Z = v[1].predict_classes(pca.inverse_transform(np.c_[xx.ravel(), yy.ravel()])) 

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)

    ax[i].contourf(xx, yy, Z, cmap=plt.cm.Paired)
    #ax[i].set_axis('off')
    # Plot also the training points
    ax[i].scatter(X_t[:, 0], X_t[:, 1], c=np.argmax(y, axis=1), cmap=plt.cm.Paired)
    ax[i].set_title(v[0])



