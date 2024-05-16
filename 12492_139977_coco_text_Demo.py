import coco_text

ct = coco_text.COCO_Text('COCO_Text.json')

ct.info()

imgs = ct.getImgIds(imgIds=ct.train, 
                    catIds=[('legibility','legible'),('class','machine printed')])

anns = ct.getAnnIds(imgIds=ct.val, 
                        catIds=[('legibility','legible'),('class','machine printed')], 
                        areaRng=[0,200])

dataDir='../../../Desktop/MSCOCO/data'
dataType='train2014'

get_ipython().magic('matplotlib inline')
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
pylab.rcParams['figure.figsize'] = (10.0, 8.0)

# get all images containing at least one instance of legible text
imgIds = ct.getImgIds(imgIds=ct.train, 
                    catIds=[('legibility','legible')])
# pick one at random
img = ct.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]

I = io.imread('%s/images/%s/%s'%(dataDir,dataType,img['file_name']))
print '/images/%s/%s'%(dataType,img['file_name'])
plt.figure()
plt.imshow(I)

# load and display text annotations
plt.imshow(I)
annIds = ct.getAnnIds(imgIds=img['id'])
anns = ct.loadAnns(annIds)
ct.showAnns(anns)

import coco_evaluation

our_results = ct.loadRes('our_results.json')

our_detections = coco_evaluation.getDetections(ct, our_results, detection_threshold = 0.5)

print 'True positives have a ground truth id and an evaluation id: ', our_detections['true_positives'][0]
print 'False positives only have an evaluation id: ', our_detections['false_positives'][0]
print 'True negatives only have a ground truth id: ', our_detections['false_negatives'][0]

our_endToEnd_results = coco_evaluation.evaluateEndToEnd(ct, our_results, detection_threshold = 0.5)

coco_evaluation.printDetailedResults(ct,our_detections,our_endToEnd_results,'our approach')

