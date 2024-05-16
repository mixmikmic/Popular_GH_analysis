import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import numpy as np
from PIL import Image, ImageOps

mnist = input_data.read_data_sets('/tmp/data', one_hot=True)

rows = 32
cols = rows
im_size = 28

sprite = np.zeros((rows*im_size, cols*im_size))

idx = -1
for i in range(rows):
    for j in range(cols):
        idx +=1
        image = mnist.test.images[idx].reshape((28,28))
        row_coord = i * 28
        col_coord = j * 28
        sprite[row_coord:row_coord + 28, col_coord:col_coord + 28] = image
        
im = Image.fromarray(sprite * 255)
im = im.convert('RGB')
im = ImageOps.invert(im)

def get_color(lbl):
    if lbl == 0: return (255, 102, 102)
    if lbl == 1: return (255, 178, 102)
    if lbl == 2: return (255, 255, 102)
    if lbl == 3: return (178, 255, 102)
    if lbl == 4: return (102, 255, 102)
    if lbl == 5: return (102, 255, 178)
    if lbl == 6: return (102, 255, 255)
    if lbl == 7: return (102, 178, 255)
    if lbl == 8: return (102, 102, 255)
    if lbl == 9: return (178, 102, 255)

labels_file = open("labels.tsv", "w")
    
# colorize
orig_color = (255,255,255)
data = np.array(im)

idx = -1
for i in range(rows):
    for j in range(cols):
        idx +=1
        row_coord = i * 28
        col_coord = j * 28
        label = np.argmax(mnist.test.labels[idx])
        labels_file.write(str(label) + "\n")
        replacement_color = get_color(label)
        r = data[row_coord:row_coord + 28, col_coord:col_coord + 28]
        r[(r == orig_color).all(axis = -1)] = replacement_color

im = Image.fromarray(data, mode='RGB')
im.save("sprite.png")
im.show()

labels_file.close()

