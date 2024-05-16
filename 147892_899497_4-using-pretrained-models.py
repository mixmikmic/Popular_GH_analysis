import numpy as np

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.applications.resnet50 import preprocess_input, decode_predictions

# img src = 'https://gfp-2a3tnpzj.stackpathdns.com/wp-content/uploads/2016/07/Dachshund-600x600.jpg'
img = load_img('dog.jpg')

img

from keras.applications.resnet50 import ResNet50

model = ResNet50(weights='imagenet')

img = load_img('dog.jpg', target_size = (224, 224))    # image size can be calibrated with target_size parameter
img

img = img_to_array(img)
print(img.shape)

img = np.expand_dims(img, axis=0)
print(img.shape)

## prediction wo preprocessing
pred_class = model.predict(img)
# print(pred_class)

# print only top 10 predicted classes
n = 10
top_n = decode_predictions(pred_class, top=n)

for c in top_n[0]:
    print(c)

img = preprocess_input(img)    # preprocess image with preprocess_input function
print(img.shape)

## prediction with preprocessing
pred_class = model.predict(img)
# print(pred_class)

n = 10
top_n = decode_predictions(pred_class, top=n)

for c in top_n[0]:
    print(c)

