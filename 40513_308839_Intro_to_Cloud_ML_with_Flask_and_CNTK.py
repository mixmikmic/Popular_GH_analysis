#load libraries
import os,sys
import pkg_resources
from flask import Flask, render_template, request, send_file
get_ipython().magic('matplotlib inline')
import matplotlib
import matplotlib.pyplot as plt
import wget
import numpy as np
from PIL import Image, ImageOps
from urllib.request import urlretrieve
import requests
from cntk import load_model, combine
from io import BytesIO, StringIO
import base64
from IPython.core.display import display, HTML
import aiohttp
import asyncio
import json
import random

print("System version: {}".format(sys.version))
print("Flask version: {}".format(pkg_resources.get_distribution("flask").version))
print("CNTK version: {}".format(pkg_resources.get_distribution("cntk").version))

def maybe_download_model(filename='ResNet_18.model'):
    if(os.path.isfile(filename)):
        print("Model %s already downloaded" % filename)
    else:
        model_name_to_url = {
        'AlexNet.model':   'https://www.cntk.ai/Models/AlexNet/AlexNet.model',
        'AlexNetBS.model': 'https://www.cntk.ai/Models/AlexNet/AlexNetBS.model',
        'VGG_16.model': 'https://www.cntk.ai/Models/Caffe_Converted/VGG16_ImageNet.model',
        'VGG_19.model': 'https://www.cntk.ai/Models/Caffe_Converted/VGG19_ImageNet.model',
        'InceptionBN.model': 'https://www.cntk.ai/Models/Caffe_Converted/BNInception_ImageNet.model',
        'ResNet_18.model': 'https://www.cntk.ai/Models/ResNet/ResNet_18.model',
        'ResNet_50.model': 'https://www.cntk.ai/Models/Caffe_Converted/ResNet50_ImageNet.model',
        'ResNet_101.model': 'https://www.cntk.ai/Models/Caffe_Converted/ResNet101_ImageNet.model',
        'ResNet_152.model': 'https://www.cntk.ai/Models/Caffe_Converted/ResNet152_ImageNet.model'
        }
        url = model_name_to_url[filename] 
        wget.download(url, out=filename)

get_ipython().run_cell_magic('time', '', "model_name = 'ResNet_152.model'\nIMAGE_MEAN = 0 # in case the CNN rests the mean for the image\nmaybe_download_model(model_name)")

def read_synsets(filename='synsets.txt'):
    with open(filename, 'r') as f:
        synsets = [l.rstrip() for l in f]
        labels = [" ".join(l.split(" ")[1:]) for l in synsets]
    return labels

labels = read_synsets()
print("Label length: ", len(labels))
print(labels[:5])

def read_image_from_file(filename):
    img = Image.open(filename)
    return img
def read_image_from_ioreader(image_request):
    img = Image.open(BytesIO(image_request.read())).convert('RGB')
    return img
def read_image_from_request_base64(image_base64):
    img = Image.open(BytesIO(base64.b64decode(image_base64)))
    return img
def read_image_from_url(url):
    img = Image.open(requests.get(url, stream=True).raw)
    return img

def plot_image(img):
    plt.imshow(img)
    plt.axis('off')
    plt.show()

imagepath = 'neko.jpg'
img_cat = read_image_from_file(imagepath)
plot_image(img_cat)

imagefile = open(imagepath, 'rb')
print(type(imagefile))
img = read_image_from_ioreader(imagefile)
plot_image(img)

imagefile = open(imagepath, 'rb')
image_base64 = base64.b64encode(imagefile.read())
print("String of %d characters" % len(image_base64))
img = read_image_from_request_base64(image_base64)
plot_image(img)

imageurl = 'https://pbs.twimg.com/profile_images/269279233/llama270977_smiling_llama_400x400.jpg'
img_llama = read_image_from_url(imageurl)
plot_image(img_llama)

get_ipython().run_cell_magic('time', '', 'z = load_model(model_name)')

def softmax(vect):
    return np.exp(vect) / np.sum(np.exp(vect), axis=0)

def get_preprocessed_image(my_image, mean_image):
    #Crop and center the image
    my_image = ImageOps.fit(my_image, (224, 224), Image.ANTIALIAS)
    #Transform the image for CNTK format
    my_image = np.array(my_image, dtype=np.float32)
    # RGB -> BGR
    bgr_image = my_image[:, :, ::-1] 
    image_data = np.ascontiguousarray(np.transpose(bgr_image, (2, 0, 1)))
    image_data -= mean_image
    return image_data

def predict(model, image, labels, number_results=5):
    img = get_preprocessed_image(image, IMAGE_MEAN)
    # Use last layer to make prediction
    arguments = {model.arguments[0]: [img]}
    result = model.eval(arguments)
    result = np.squeeze(result)
    prob = softmax(result)
    # Sort probabilities 
    prob_idx = np.argsort(result)[::-1][:number_results]
    pred = [labels[i] for i in prob_idx]
    return pred
 

resp = predict(z, img_llama, labels, 2)
print(resp)
resp = predict(z, img_cat, labels, 3)
print(resp)
resp = predict(z, read_image_from_url('http://www.awf.org/sites/default/files/media/gallery/wildlife/Hippo/Hipp_joe.jpg'), labels, 5)
print(resp)

get_ipython().run_cell_magic('bash', '--bg ', '/home/my-user/anaconda3/envs/my-cntk-env/bin/python /home/my-user/sciblog_support/Intro_to_Machine_Learning_API/cntk_api.py')

res = requests.get('http://127.0.0.1:5000/')
display(HTML(res.text))

headers = {'Content-type':'application/json'}
data = {'param':'1'}
res = requests.post('http://127.0.0.1:5000/api/v1/classify_image', data=json.dumps(data), headers=headers)
print(res.text)

get_ipython().run_cell_magic('time', '', "imageurl = 'https://pbs.twimg.com/profile_images/269279233/llama270977_smiling_llama_400x400.jpg'\ndata = {'url':imageurl}\nres = requests.post('http://127.0.0.1:5000/api/v1/classify_image', data=json.dumps(data), headers=headers)\nprint(res.text)")

get_ipython().run_cell_magic('time', '', "imagepath = 'neko.jpg'\nimage_request = open(imagepath, 'rb')\nfiles_local = {'image': image_request}\nres = requests.post('http://127.0.0.1:5000/api/v1/classify_image', files=files_local)\nprint(res.text)")

server_name = 'http://the-name-of-your-server'
port = 5000

root_url = '{}:{}'.format(server_name, port)

res = requests.get(root_url)
display(HTML(res.text))

end_point = root_url + '/api/v1/classify_image' 
#print(end_point)

get_ipython().run_cell_magic('time', '', "imageurl = 'https://pbs.twimg.com/profile_images/269279233/llama270977_smiling_llama_400x400.jpg'\ndata = {'url':imageurl}\nheaders = {'Content-type':'application/json'}\nres = requests.post(end_point, data=json.dumps(data), headers=headers)\nprint(res.text)")

get_ipython().run_cell_magic('time', '', "imagepath = 'neko.jpg'\nimage_request = open(imagepath, 'rb')\nfiles = {'image': image_request}\nres = requests.post(end_point, files=files)\nprint(res.text)")

# Get hippo
hippo_url = "http://www.awf.org/sites/default/files/media/gallery/wildlife/Hippo/Hipp_joe.jpg"

fname = urlretrieve(hippo_url, "bhippo.jpg")[0]
img_bomb = read_image_from_file(fname)
plot_image(img_bomb)

NUM = 100
concurrent = 10

def gen_variations_of_one_image(num, filename):
    out_images = []
    imagefile = open(filename, 'rb')
    img = Image.open(BytesIO(imagefile.read())).convert('RGB')
    img = ImageOps.fit(img, (224, 224), Image.ANTIALIAS)
    # Flip the colours for one-pixel
    # "Different Image"
    for i in range(num):
        diff_img = img.copy()
        rndm_pixel_x_y = (random.randint(0, diff_img.size[0]-1), 
                          random.randint(0, diff_img.size[1]-1))
        current_color = diff_img.getpixel(rndm_pixel_x_y)
        diff_img.putpixel(rndm_pixel_x_y, current_color[::-1])
        # Turn image into IO
        ret_imgio = BytesIO()
        diff_img.save(ret_imgio, 'PNG')
        out_images.append(ret_imgio.getvalue())
    return out_images

get_ipython().run_cell_magic('time', '', "# Save same file multiple times in memory as IO\nimages = gen_variations_of_one_image(NUM, fname)\nurl_list = [[end_point, {'image':pic}] for pic in images]")

def handle_req(data):
    return json.loads(data.decode('utf-8'))
 
def chunked_http_client(num_chunks, s):
    # Use semaphore to limit number of requests
    semaphore = asyncio.Semaphore(num_chunks)
    @asyncio.coroutine
    # Return co-routine that will work asynchronously and respect
    # locking of semaphore
    def http_get(dta):
        nonlocal semaphore
        with (yield from semaphore):
            url, img = dta
            response = yield from s.request('post', url, data=img)
            body = yield from response.content.read()
            yield from response.wait_for_close()
        return body
    return http_get

    
def run_experiment(urls, _session):
    http_client = chunked_http_client(num_chunks=concurrent, s=_session)
    
    # http_client returns futures, save all the futures to a list
    tasks = [http_client(url) for url in urls]
    dfs_route = []
    
    # wait for futures to be ready then iterate over them
    for future in asyncio.as_completed(tasks):
        data = yield from future
        try:
            out = handle_req(data)
            dfs_route.append(out)
        except Exception as err:
            print("Error {0}".format(err))
    return dfs_route

get_ipython().run_cell_magic('time', '', "# Expect to see some 'errors' meaning requests are expiring on 'queue'\n# i.e. we can't increase concurrency any more\nwith aiohttp.ClientSession() as session:  # We create a persistent connection\n    loop = asyncio.get_event_loop()\n    complete_responses = loop.run_until_complete(run_experiment(url_list, session)) ")

print("Number of sucessful queries: {} of {}".format(len(complete_responses), NUM))
print(complete_responses[:5])

