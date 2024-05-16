import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications

vggmodel = applications.VGG16(include_top=False, weights='imagenet')

datagen = ImageDataGenerator(rescale=1./255)

# you are passing your images through the pretrained vgg network
# all the way up until the fully connected layers

generator = datagen.flow_from_directory('data/train',
                                       target_size=(150, 150),
                                       batch_size=16,
                                       class_mode=None,
                                       shuffle=False)

bottleneck_features_train = vggmodel.predict_generator(generator, 2000/16)

np.save(open('bottleneck_features_train.npy','wb'), bottleneck_features_train)

# reload the file that has been pre created and predicted on

# preload the training data and create the true output labels

train_data = np.load(open('bottleneck_features_train.npy','rb'))

train_labels = np.array([0]*1000 + [1]*1000)

train_data.shape

# test data to check your accuracy on

validation_data = np.load(open('bottleneck_features_validation.npy','rb'))

validation_labels = np.array([0]*400 + [1]*400)

# create your own fully connected network to make predictions on dog vs cat

model = Sequential()

model.add(Flatten(input_shape=(4,4,512)))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# define the loss function and the gradient descent function

model.compile(optimizer='rmsprop',
             loss='binary_crossentropy',
             metrics=['accuracy'])

# train on training data

model.fit(train_data, train_labels,
         epochs=30,
         batch_size=16,
         validation_data=(validation_data, validation_labels))





import matplotlib.pyplot as plt
from scipy.misc import imresize

get_ipython().magic('matplotlib inline')

# read in your image
img = plt.imread('dog.jpg')

# show your image
plt.imshow(img)

# resize my image
img = imresize(img, (150,150,3))

plt.imshow(img)

img.shape

# tells you there is one image in this set
img = np.expand_dims(img, axis=0)

img.shape

convolved_img = vggmodel.predict(img,1)

convolved_img.shape

new_dn_model = Sequential()

new_dn_model.add(Flatten(input_shape=(4,4,512)))
new_dn_model.add(Dense(256, activation='relu'))
new_dn_model.add(Dropout(0.5))
new_dn_model.add(Dense(1, activation='sigmoid'))

new_dn_model.load_weights('bottleneck_fc_model.h5')

new_dn_model.predict_classes(convolved_img)



