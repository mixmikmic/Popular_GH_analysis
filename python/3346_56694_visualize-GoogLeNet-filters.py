# imports and basic notebook setup
from cStringIO import StringIO
import numpy as np
import os,re,random
import scipy.ndimage as nd
import PIL.Image
import sys
from IPython.display import clear_output, Image, display
from scipy.misc import imresize

pycaffe_root = "/your/path/here/caffe/python" # substitute your path here
sys.path.insert(0, pycaffe_root)
import caffe

model_name="GoogLeNet"
model_path = '/your/path/here/caffe_models/bvlc_googlenet/' # substitute your path here
# modified deploy.prototxt, switched relus to leaky relus
net_fn   = './googlenet_deploy_mod.prototxt'
param_fn = model_path + 'bvlc_googlenet.caffemodel'
means = np.float32([104.0, 117.0, 123.0])

#caffe.set_mode_gpu() # uncomment this if gpu processing is available

net = caffe.Classifier(net_fn, param_fn,
                       mean = means, # ImageNet mean, training set dependent
                       channel_swap = (2,1,0)) # the model has channels in BGR order instead of RGB

# a couple of utility functions for converting to and from Caffe's input image layout
def preprocess(net, img):
    return np.float32(np.rollaxis(img, 2)[::-1]) - net.transformer.mean['data']
def deprocess(net, img):
    return np.dstack((img + net.transformer.mean['data'])[::-1])

def blur(img, sigma):
    if sigma > 0:
        img[0] = nd.filters.gaussian_filter(img[0], sigma, order=0)
        img[1] = nd.filters.gaussian_filter(img[1], sigma, order=0)
        img[2] = nd.filters.gaussian_filter(img[2], sigma, order=0)
    return img

def showarray(a, f, fmt='jpeg'):
    a = np.uint8(np.clip(a, 0, 255))
    f = StringIO()
    PIL.Image.fromarray(a).save(f, fmt)
    display(Image(data=f.getvalue()))

def make_step(net, step_size=1.5, end='inception_4c/output', clip=True, focus=None, sigma=None):
    '''Basic gradient ascent step.'''

    src = net.blobs['data'] # input image is stored in Net's 'data' blob

    dst = net.blobs[end]
    net.forward(end=end)
    
    one_hot = np.zeros_like(dst.data)
    filter_shape = dst.data.shape
    if len(filter_shape) > 2:
        # backprop only activation in middle of filter
        one_hot[0,focus,(filter_shape[2]-1)/2,(filter_shape[3]-1)/2] = 1.
    else:
        one_hot.flat[focus] = 1.
    dst.diff[:] = one_hot
    
    net.backward(start=end)
    g = src.diff[0]
    
    src.data[:] += step_size/np.abs(g).mean() * g

    if clip:
        bias = net.transformer.mean['data']
        src.data[:] = np.clip(src.data, -bias, 255-bias) 
        
    src.data[0] = blur(src.data[0], sigma)
    
    dst.diff.fill(0.)

def deepdraw(net, base_img, octaves, random_crop=True, visualize=True, focus=None,
    clip=True, **step_params):
    
    # prepare base image
    image = preprocess(net, base_img) # (3,224,224)
    
    # get input dimensions from net
    w = net.blobs['data'].width
    h = net.blobs['data'].height
    
    print "starting drawing"
    src = net.blobs['data']
    src.reshape(1,3,h,w) # resize the network's input image size
    for e,o in enumerate(octaves):
        if 'scale' in o:
            # resize by o['scale'] if it exists
            image = nd.zoom(image, (1,o['scale'],o['scale']))
        _,imw,imh = image.shape
        
        # select layer
        layer = o['layer']
        
        for i in xrange(o['iter_n']):
            if imw > w:
                if random_crop:
                    # randomly select a crop 
                    #ox = random.randint(0,imw-224)
                    #oy = random.randint(0,imh-224)
                    mid_x = (imw-w)/2.
                    width_x = imw-w
                    ox = np.random.normal(mid_x, width_x*0.3, 1)
                    ox = int(np.clip(ox,0,imw-w))
                    mid_y = (imh-h)/2.
                    width_y = imh-h
                    oy = np.random.normal(mid_y, width_y*0.3, 1)
                    oy = int(np.clip(oy,0,imh-h))
                    # insert the crop into src.data[0]
                    src.data[0] = image[:,ox:ox+w,oy:oy+h]
                else:
                    ox = (imw-w)/2.
                    oy = (imh-h)/2.
                    src.data[0] = image[:,ox:ox+w,oy:oy+h]
            else:
                ox = 0
                oy = 0
                src.data[0] = image.copy()

            sigma = o['start_sigma'] + ((o['end_sigma'] - o['start_sigma']) * i) / o['iter_n']
            step_size = o['start_step_size'] + ((o['end_step_size'] - o['start_step_size']) * i) / o['iter_n']
            
            make_step(net, end=layer, clip=clip, focus=focus, 
                      sigma=sigma, step_size=step_size)
            
            if visualize:
                vis = deprocess(net, src.data[0])
                if not clip: # adjust image contrast if clipping is disabled
                    vis = vis*(255.0/np.percentile(vis, 99.98))
                if i % 1 == 0:
                    showarray(vis,"./filename"+str(i)+".jpg")
            
            if i % 10 == 0:
                print 'finished step %d in octave %d' % (i,e)
            
            # insert modified image back into original image (if necessary)
            image[:,ox:ox+w,oy:oy+h] = src.data[0]
        
        print "octave %d image:" % e
        showarray(deprocess(net, image),"./octave_"+str(e)+".jpg")
            
    # returning the resulting image
    return deprocess(net, image)

octaves = [
    {
        'layer':'inception_4c/output',
        'iter_n':200,
        'start_sigma':2.5,
        'end_sigma':1.1,
        'start_step_size':12.,
        'end_step_size':10.,
    },
    {
        'layer':'inception_4c/output',
        'iter_n':100,
        'start_sigma':1.1,
        'end_sigma':0.78*1.1,
        'start_step_size':10.,
        'end_step_size':8.
    },
    {
        'layer':'inception_4c/output',
        'scale':1.05,
        'iter_n':100,
        'start_sigma':0.78*1.1,
        'end_sigma':0.78,
        'start_step_size':8.,
        'end_step_size':6.
    },
    {
        'layer':'inception_4c/output',
        'scale':1.05,
        'iter_n':50,
        'start_sigma':0.78*1.1,
        'end_sigma':0.40,
        'start_step_size':6.,
        'end_step_size':1.5
    },
    {
        'layer':'inception_4c/output',
        'scale':1.05,
        'iter_n':25,
        'start_sigma':0.4,
        'end_sigma':0.3,
        'start_step_size':1.5,
        'end_step_size':0.5
    }
]

# get original input size of network
original_w = net.blobs['data'].width
original_h = net.blobs['data'].height
# the background color of the initial image
background_color = np.float32([250.0, 250.0, 250.0])
# generate initial random image
gen_image = np.random.normal(background_color, 8, (original_w, original_h, 3))

# which filter in layer to visualize (conv5 has 512 filters)
imagenet_class = 411

# generate class visualization via octavewise gradient ascent
gen_image = deepdraw(net, gen_image, octaves, focus=imagenet_class, 
                 random_crop=True, visualize=False)

# save image
#img_fn = '_'.join([model_name, "deepdraw", str(imagenet_class)+'.png'])
#PIL.Image.fromarray(np.uint8(gen_image)).save('./' + img_fn)



