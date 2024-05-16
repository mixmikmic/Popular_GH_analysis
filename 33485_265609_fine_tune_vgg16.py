from keras.applications.vgg16 import VGG16

# arguments see https://keras.io/applications/#vgg16
model_vgg16 = VGG16(include_top=True, weights='imagenet')

print(model_vgg16.summary())

print model_vgg16.layers[1].get_weights()





