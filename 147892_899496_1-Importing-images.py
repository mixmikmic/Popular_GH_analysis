import numpy as np
import matplotlib.pyplot as plt

from io import BytesIO
from PIL import Image
from urllib.request import urlopen
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

# url of an image
url = 'https://gfp-2a3tnpzj.stackpathdns.com/wp-content/uploads/2016/07/Dachshund-600x600.jpg'

image =  BytesIO(urlopen(url).read())

image = Image.open(image)
image

# images can be resized using resize() method
image = image.resize((100,100))
image

# images can be converted to arrays using img_to_array() method
image_arr = img_to_array(image)
print(image_arr.shape)

# by reshaping the image array, we could get array with rank =4 
image_arr = image_arr.reshape((1,) + image_arr.shape) 
print(image_arr.shape)    # first element in shape means that we have one image data instance

url1 = 'http://cdn2-www.dogtime.com/assets/uploads/2011/01/file_23020_dachshund-dog-breed.jpg'
url2 = 'http://lovedachshund.com/wp-content/uploads/2016/06/short-haired-dachshund.jpg'
url3 = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQTavNocbsgwxukPI9eO3jgNaxP_DupqzBqE2M1oQi_GzdMHIZ-'

urls = [url1, url2, url3]

for url in urls:
    image =  BytesIO(urlopen(url).read())
    image = Image.open(image).resize((100, 100), Image.ANTIALIAS)
    image = img_to_array(image)
    image = image.reshape((1,) + image.shape) 
    print(image.shape)
    image_arr = np.concatenate((image_arr, image), axis = 0)

image_arr.shape

# visualizing imported images
num = len(image_arr)
for i in range(num):
    ax = plt.subplot(1, num, i+1 )
    plt.imshow(image_arr[i])
    
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()    

