get_ipython().magic('matplotlib inline')
from refer import REFER
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt

data_root = './data'  # contains refclef, refcoco, refcoco+, refcocog and images
dataset = 'refcoco'
splitBy = 'unc'
refer = REFER(data_root, dataset, splitBy)

# print stats about the given dataset
print 'dataset [%s_%s] contains: ' % (dataset, splitBy)
ref_ids = refer.getRefIds()
image_ids = refer.getImgIds()
print '%s expressions for %s refs in %s images.' % (len(refer.Sents), len(ref_ids), len(image_ids))

print '\nAmong them:'
if dataset == 'refclef':
    if splitBy == 'unc':
        splits = ['train', 'val', 'testA', 'testB', 'testC']
    else:
        splits = ['train', 'val', 'test']
elif dataset == 'refcoco':
    splits = ['train', 'val', 'test']
elif dataset == 'refcoco+':
    splits = ['train', 'val', 'test']
elif dataset == 'refcocog':
    splits = ['train', 'val']  # we don't have test split for refcocog right now.
    
for split in splits:
    ref_ids = refer.getRefIds(split=split)
    print '%s refs are in split [%s].' % (len(ref_ids), split)

# randomly sample one ref
ref_ids = refer.getRefIds()
ref_id = ref_ids[np.random.randint(0, len(ref_ids))]
ref = refer.Refs[ref_id]
print 'ref_id [%s] (ann_id [%s])' % (ref_id, refer.refToAnn[ref_id]['id'])
# show the segmentation of the referred object
plt.figure()
refer.showRef(ref, seg_box='seg')
plt.show()

# or show the bounding box of the referred object
refer.showRef(ref, seg_box='box')
plt.show()

# let's look at the details of each ref
for sent in ref['sentences']:
    print 'sent_id[%s]: %s' % (sent['sent_id'], sent['sent'])

