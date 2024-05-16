import sys
sys.path.append("attalos/")
from attalos.evaluation.evaluation import Evaluation
from oct2py import octave
octave.addpath('attalos/attalos/evaluation/')
import json
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from IPython.core.display import display
from IPython.core.pylabtools import figsize
from scripts import load_combine_json_dir, print_list_to_columns

jsondir = "/data/fs4/datasets/espgame/ESP-ImageSet"
imgdir = "/data/fs4/datasets/espgame/ESP-ImageSet/test_images"
wordvecs_dir = "/data/fs4/teams/attalos/wordvecs"

with open(os.path.join(jsondir,"esp_test_imageset_tags_predictions.json")) as jdata:
    predicts = json.load(jdata)

with open(os.path.join(jsondir,"esp_test_imageset_tags.json")) as jdata:
    truthdicts = json.load(jdata)

fname = np.random.choice(predicts.keys())
print fname
truthlbls = truthdicts[fname]
predlbls = predicts[fname]

img = Image.open(os.path.join(imgdir, fname))
display(img)

print("*" * 50 + "\nGround Truth labels:\n")
print_list_to_columns(truthlbls)

print("*" * 50 + "\nPredicted labels:\n")
print_list_to_columns(predlbls[0:10],items_per_row=6)

glove_vector_file = os.path.join(wordvecs_dir, "glove.6B.200d.txt")

f=open(glove_vector_file,'r')

glove_vec_dic = {}
for i in f.read().splitlines():
    t=i.split()
    tv=[]
    for j in t[1:]:
        tv.append(float(j))
    glove_vec_dic[t[0]]=tv

f.close()

esp_dir = "/data/fs4/datasets/espgame"
data = np.load(os.path.join(esp_dir,"espgame-inria.npz"))
esp_vocab = data["D"] # ESP vocabulary
yTe = data ["yTe"] # ESP onehot matrix
testlist = data['testlist'] # ESP test images

esp_vocab_vec = np.zeros((len(esp_vocab),len(glove_vec_dic.values()[0])))

for i in range(len(esp_vocab)):
    esp_vocab_vec[i]=glove_vec_dic[esp_vocab[i]]

with open("/data/fs4/home/justinm/andrew-attalos/andreww/VG-object-regions-dicts.json.txt") as jdata:
    vgg_json_vocab = json.load(jdata)

vgg_vocab=[]
for i in vgg_json_vocab['token_to_idx'].keys():
    vgg_vocab.append(str(i))

temp_vgg_vocab = []
for i in vgg_vocab:
    if i in glove_vec_dic.keys():
        temp_vgg_vocab.append(i)
    #if i % 500 == 0:
    #    print i, ' out of ', len(vgg_vocab)
vgg_vocab_not_in_glove = list(set(vgg_vocab)-set(temp_vgg_vocab))[:]

vgg_vocab_vec = np.zeros((len(vgg_vocab),len(glove_vec_dic.values()[0])))
print len(vgg_vocab)
print len(temp_vgg_vocab)
print len(vgg_vocab)-len(temp_vgg_vocab)
print vgg_vocab_vec.shape

for i in range(len(vgg_vocab)):
    if vgg_vocab[i] in vgg_vocab_not_in_glove:
        vgg_vocab_vec[i,:]=0
    else:
        vgg_vocab_vec[i]=glove_vec_dic[vgg_vocab[i]]

esp_not_in_glove = []
esp_not_in_vgg = []
for i in esp_vocab:
    if i not in vgg_vocab:
        esp_not_in_vgg.append(i)
        print i, " not in VG vocab"
    if i not in glove_vec_dic.keys():
        esp_not_in_glove.append(i)
        print "\t",i, " not in GloVe"

# Get the word not in VG and index of the image in testlist
esp_gt_not_in_vgg={}
for i in esp_not_in_vgg:
    idxword = np.argwhere(esp_vocab==i)
    idxsimg = np.nonzero(yTe[:,idxword])[0]
    if len(idxsimg) != 0:
        esp_gt_not_in_vgg[i] = idxsimg

# print ESP word not in VG and the image index
for i in esp_gt_not_in_vgg.keys():
    print i, esp_gt_not_in_vgg[i]

# print words in VG but not in ESP
vgg_vocab_not_in_glove = list(set(vgg_vocab)-set(temp_vgg_vocab))[:]
print len(vgg_vocab_not_in_glove), " words in VG but not in GloVe"
print vgg_vocab_not_in_glove

# convert ESP word vectors to unit vectors
esp_vocab_vec_norm = np.divide(esp_vocab_vec.T,np.linalg.norm(esp_vocab_vec,axis=1)).T

# convert VG word vectors to unit vectors. Do to the VG words not in GloVe, those vectors have length 0.

vgg_norm = np.linalg.norm(vgg_vocab_vec,axis=1)
vgg_vocab_vec_norm = vgg_vocab_vec.copy()

# Have to convert each word vector one at a time do to some word vector lengths of 0
for i in xrange(len(vgg_norm)):
    if vgg_norm[i] != 0:
        vgg_vocab_vec_norm[i] = np.divide(vgg_vocab_vec[i],vgg_norm[i])

# create the correlation matrix between the ESP word vectors and VG word Vectors
esp_vgg_corr = np.dot(esp_vocab_vec_norm,vgg_vocab_vec_norm.T)

# Testing the correlation of the same word has dot product of 1
print 'airplane' in esp_vocab
print 'airplane' in vgg_vocab
print esp_vgg_corr[np.argwhere(esp_vocab==('airplane'))[0][0],vgg_vocab.index('airplane')]

# reduce predicted tags to the top 5 without repetitive words and <UNK> tokens
# the predicted tags are in order from most confidence to least
reduced_predicts={}
top_n_words = 5
lessthan5 = 0
print "number of test images = ",len(predicts.keys())
for i in predicts.keys():
    words = predicts[i]
    newl = [] # new list of top n predicted words 
    norepeats = [] # complete list of words without repeated words and <UNK> tokens
    for j in words:
        if len(newl) >= top_n_words and j not in norepeats:
            norepeats.append(j)
        elif j != "<UNK>" and j not in newl:# and j in vocab:
            newl.append(j)
            norepeats.append(j)
    reduced_predicts[i] = newl
    if len(newl) < top_n_words:
        lessthan5+=1
        #print newl, len(newl)
        #print norepeats
print lessthan5, "/", len(predicts.keys()), " have less than 5 predicted words that are in the ESP-Games dictionay"

# list predicted tags that are not in ESP vocab
no_pred_words_in_esp_dict = {}
no_pred_words_in_glv_dict = {}
for i in reduced_predicts.keys():
    no_pred_words_in_esp = []
    no_pred_words_in_glv = []
    for j in reduced_predicts[i]:
        if str(j) not in esp_vocab:
            no_pred_words_in_esp.append(j)
        if str(j) in vgg_vocab_not_in_glove:
            no_pred_words_in_glv.append(j)
    if len(no_pred_words_in_esp)>0:
        no_pred_words_in_esp_dict[i] = no_pred_words_in_esp
        print i,no_pred_words_in_esp," predictions not in esp vocab"
    if len(no_pred_words_in_glv)>0:
        no_pred_words_in_glv_dict[i] = no_pred_words_in_esp
        print i,no_pred_words_in_glv," predictions not in GloVe dictionary"

#print len(no_pred_words_in_esp_dict.keys())," number of images that have a top 5 tag not in esp vocab"
#print len(no_pred_words_in_glv_dict.keys())," number of images that have a top 5 tag not in glove dictionary"

# Create onehot matrix of predicted tags. 
# Also translating VG words not in ESP vocab to the highest correlated word in ESP vocab
predict_arr = np.zeros(yTe.shape,dtype=np.int)
x_corpa_word_map={} # image as key, value is [VG word, correlated ESP word, dot product of words]
for i in reduced_predicts.keys():
    idximg = np.argwhere(testlist==i)[0][0]
    for j in reduced_predicts[i]:
        wordmap=[]
        idxwrd = np.argmax(esp_vgg_corr[:,vgg_vocab.index(j)])
        predict_arr[idximg,idxwrd] = 1
        if str(j) != esp_vocab[idxwrd]:
            wordmap.append([str(j),
                            esp_vocab[idxwrd],
                            esp_vgg_corr[np.argmax(esp_vgg_corr[:,vgg_vocab.index(j)]),vgg_vocab.index(j)]])
        if esp_vgg_corr[np.argmax(esp_vgg_corr[:,vgg_vocab.index(j)]),vgg_vocab.index(j)] == 0:
            print 'help, vg word has no correlation with a word in ESP \t',wordmap[-1] # fortunatelly this did not happen
    if len(wordmap) != 0:
        x_corpa_word_map[i]=wordmap

# Reduce repeated word correlations
reduced_x_corpa_word_map = []
for i in x_corpa_word_map.keys():
    for j in x_corpa_word_map[i]:
        if j not in reduced_x_corpa_word_map:
            reduced_x_corpa_word_map.append(j)
print len(reduced_x_corpa_word_map), "VG words had to be correlated to an ESP word"

# print the mapping from VG word to the highest correlated ESP word
print "[predicted word in vg, word with highest correlation to esp word, correlation]"
for i in reduced_x_corpa_word_map[:]:
    print i

# print ESP word not in VG and the image index
num_tags_not_vgg_in_esp_gt = 0
for i in esp_gt_not_in_vgg.keys():
    print i, esp_gt_not_in_vgg[i]
    num_tags_not_vgg_in_esp_gt += len(esp_gt_not_in_vgg[i])

print num_tags_not_vgg_in_esp_gt, " number of times an ESP word not in VG but was used in the ESP test images"
print np.sum(yTe), " number of ESP ground truth tags"
print num_tags_not_vgg_in_esp_gt/np.sum(yTe) * 100, "%"

# Evaluate the ground truth tags (yTe) to the top n predicted tags (predict_arr)
[precision,recall,f1] = octave.evaluate(yTe.T, predict_arr.T, 5)
print("Precision: {0:0.3f}".format(precision))
print("Recall: {0:0.3f}".format(recall))
print("F-1: {0:0.3f}".format(f1))


intersection = 0
union = 0
recall_denom = 0
'''
for fname in gt_set.iterkeys():
    a = gt_set[fname]
    b = prediction_set[fname]
    intersection += len(a.intersection(b))
    union += len(a.union(b))
    recall_denom += len(a)
'''

################################################

for fname in truthdicts.keys():
    a = set(truthdicts[fname])
    b = set(reduced_predicts[fname])
    intersection += len(a.intersection(b))
    union += len(a.union(b))
    recall_denom += len(a)

intersection=float(intersection)

###############################################

precision = intersection / union
recall = intersection / recall_denom
f1 = 2 * (precision * recall) / (precision + recall)
print("Precision: {0:0.3f}".format(precision))
print("Recall: {0:0.3f}".format(recall))
print("F-1: {0:0.3f}".format(f1))

