from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import sys
sys.path.append("attalos/")
import json
import h5py
import numpy as np
from PIL import Image
from oct2py import octave
from scripts import loaders
from collections import Counter
from IPython.core.display import display
#from attalos.evaluation.evaluation import Evaluation
from scripts import load_combine_json_dir, print_list_to_columns

octave.addpath('../')

# The json file from feedforward 
json_dir = "densecap/predictions/one-hot/"

# directory with all source images
imgdir = "visual-genome/images/"

# preprocessed training data
object_hd5 = 'VG-object-regions.h5'
object_dict = 'VG-object-regions-dicts.json'

print("Loading predictions (This will take some time)...")
results = load_combine_json_dir(json_dir, loaders.load_output_json)
print("Done!")

# simple JSON data. Holds index mappings for images and word tokens
gt_map = loaders.json_load(object_dict)

# This HDF5 file is large (127 GB). We only want labels. We'll use the 
# json data to complete the map
with h5py.File(object_hd5, 'r') as hf:    
    # labels array 
    labels = np.array(hf.get('labels'))
    
    # We will use these indices to take slices of labels. 
    # The mapping is filename_to_idx -- img_to_first_box, img_to_last_box -- labels
    img_to_first_box = np.array(hf.get('img_to_first_box'))
    img_to_last_box = np.array(hf.get('img_to_last_box'))

def get_corresponding_labels(filename, reference_dict, labels_arr, first_box, last_box):
    """This function will do the mapping from filename to label
    
    Parameters
    ----------
    filename : str
        name of an image file. Just the os.path.basename of the file. e.g. '55.jpg'
    reference_dict : dict
        object regions dict used for model training
    labels_arr : numpy.array
        training lables array
    first_box : numpy.array
        img_to_first_box array from training data
    last_box : numpy.array
        img_to_last_box array from training data
    
    Returns
    -------
    numpy.array with label IDs
    """
    
    idx = reference_dict['filename_to_idx'][filename] - 1
    slice_start = first_box[idx]
    slice_end = last_box[idx]
    return labels[slice_start:slice_end]

n_gt_images = len(gt_map['idx_to_filename'].items())
n_tokens = len(gt_map['token_to_idx'].items())
n_imgs = len(results.keys())

# model predicts 1000 boxes per image
n_pred_boxes = n_imgs * 1000
print("Tokens: {0:,}\nPrediction Images: {1:,}\nPredicted Boxes: {3:,}\n\nGround Truth Images: {2:,}".format(n_tokens, n_imgs, n_gt_images, n_pred_boxes))

# snag a random filename 
fname = np.random.choice(gt_map['filename_to_idx'].keys())

# run through the mapping function to get the labels
label_codes = get_corresponding_labels(fname, gt_map, labels, img_to_first_box, img_to_last_box)

# open the image for display
img = Image.open(os.path.join(imgdir, fname))
display(img)

# Collect all terms and combine
terms = []
rows, cols = label_codes.shape
for row in xrange(rows):
    obj = ''
    words = []
    for col in xrange(cols):
        # The word mapping stores the token IDs as strings
        val_num = label_codes[row, col]
        val = str(val_num)
        
        if (val_num > 0) and gt_map['idx_to_token'].has_key(val):
            words.append(gt_map['idx_to_token'][val])
        terms.append(" ".join(words))
        
print("*" * 50 + "\nGround Truth labels:\n")

# function to print columns instead of one long list
print_list_to_columns(list(set(terms)))

pred_labels = list(set(map(lambda x : " ".join(x['names']), results[fname])))
print("*" * 50 + "\nPredicted labels:\n")
print_list_to_columns(pred_labels)

def get_tokens_and_score(bbox_predictions, token_to_idx):
    """This function will gather the unique tokens and 
    the best score assigned to each
    
    Parameters
    ----------
    bbox_predictions : dict
        dictionary with predictions from DenseCap. Each 
        item should have a 'score' and 'names'
    token_to_idx : dict
        mapping of words to an index/unique number
        
    Returns
    --------
    Dictionary with {term: score} mappings, dictionary 
    with missed words and counts {word, count}
    """
    out = {}
    missed = Counter()
    for item in bbox_predictions:
        name = item['names']
        
        # each name should be a list with 1 item 
        err = "Too maney tokens! Should have 1 but this one has {0}".format(len(name))
        assert len(name) == 1, err
        
        name = name[0]
        
        # attempt to squish the token into existing vocab
        # found examples of 'power line' but the existing 
        # vocab only had 'powerline'
        name_stripped = name.replace(' ', '')
        
        if token_to_idx.has_key(name):
            # 1-indexing to 0-indexing densecap/preprocessing.py 44
            name_idx = token_to_idx[name] - 1
    
        elif token_to_idx.has_key(name_stripped):
            # 1-indexing to 0-indexing densecap/preprocessing.py 44
            name_idx = token_to_idx[name_stripped] - 1 
            
        else:
            # This keeps track of the terms out of 
            # training vocabulary. 
            missed[name] += 1
            continue 
        
        score = item['score']
        if out.has_key(name_idx):
            cur_score = out[name_idx]
            if score > cur_score:
                out[name_idx] = score
        else:
            out[name_idx] = score
    return out, missed  

# array needs to be Images X Tokens
prediction_arr = np.zeros(shape=(n_gt_images, n_tokens), dtype=np.float32)

# dictionary to hold the set of unique word predictions
# we'll use this later for precision and recall numbers 
prediction_set = {}

# get the best score for each unique token
for fname, preds in results.iteritems():
    if gt_map['filename_to_idx'].has_key(fname):
        
        # 1-indexing to 0-indexing densecap/preprocessing.py 44
        idx = gt_map['filename_to_idx'][fname] - 1 
        
        scores, _ = get_tokens_and_score(preds, gt_map['token_to_idx'])
        
        # collect the tokens and scores
        tokens = []
        
        # indexing accounted for in get_tokens_and_score(...) functions
        for word_idx, score in scores.iteritems():
            prediction_arr[idx, word_idx] =  score
            tokens.append((score, word_idx))
        
        prediction_set[fname] = tokens

gt_array = np.zeros_like(prediction_arr)

gt_set = {}

# Loop over the ground truth map to fill in the 
# ground truth array 
for fname, idx in gt_map['filename_to_idx'].iteritems():
    gt_label_arr = get_corresponding_labels(fname, gt_map,                                             labels, img_to_first_box,                                             img_to_last_box)
    # need to go through each box and find non-zero tokens
    rows, cols = gt_label_arr.shape
    
    # 1-indexing to 0-indexing densecap/preprocessing.py 44
    idx -= 1
    
    tokens = set()
    for row in xrange(rows):
        vals = []
        for col in xrange(cols):
            val = gt_label_arr[row, col]
            if val > 0:
                # 1-indexing to 0-indexing densecap/preprocessing.py 44
                val -= 1
                vals.append(val)
        # We only trained on the last word of the truth, so only use the last word
        try:
            final_val = vals[-1]
        except IndexError:
            continue
        tokens.add(final_val)
        # To one-hot
        gt_array[idx, final_val] = 1
    gt_set[fname] = tokens

def get_top_n(predictions, n=5):
    sorted_preds = sorted(predictions)
    ret_val = []
    for score, word in reversed(sorted_preds):
        # Only take N items
        if len(ret_val) >= n:
            break
        # Do not take duplicates
        if word in ret_val:
            continue
        else:
            ret_val.append(word)
    return ret_val

intersection = 0
union = 0
recall_denom = 0

for fname in gt_set.iterkeys():
    a = gt_set[fname]
    b = prediction_set[fname]
    b = get_top_n(b, n=5)
    intersection += len(a.intersection(b))
    union += len(a.union(b))
    recall_denom += len(a)

precision = intersection / union
recall = intersection / recall_denom
f1 = 2 * (precision * recall) / (precision + recall)

print("Precision: {0:0.3f}".format(precision))
print("Recall: {0:0.3f}".format(recall))
print("F-1: {0:0.3f}".format(f1))

