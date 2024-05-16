from keras import applications
from keras.models import Sequential, Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input, decode_predictions
from keras.models import Model
model = VGG16(weights='imagenet', include_top=True)
model.summary()
#predicting for any new image based on the pre-trained model
# Loading Image

import numpy as np
from keras.preprocessing import image
img = image.load_img('horse.jpg', target_size=(224, 224))
img = image.img_to_array(img)
img = np.expand_dims(img, axis=0)
img=preprocess_input(img)
# Predict the output
preds = model.predict(img)
# decode the predictions
pred_class = decode_predictions(preds, top=3)[0][0]
print('Predicted Class: %s' %pred_class[1])
print('Confidance: %s'% pred_class[2])
#Predicted Class: hartebeest
#Confidance: 0.964784
#ResNet50 and InceptionV3 models can be easily utilized for prediction/classification of new images.
from keras.applications import ResNet50
model = ResNet50(weights='imagenet' , include_top=True)
model.summary()
# create the base pre-trained model
from keras.applications import InceptionV3
model = InceptionV3(weights='imagenet')
model.summary()



