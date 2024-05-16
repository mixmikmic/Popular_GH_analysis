import os

import numpy as np

import datetime
t_start=datetime.datetime.now()

import pickle

data_path = './data/Flickr30k'

output_dir = './data/cache'

output_filepath = os.path.join(output_dir, 
                                'CAPTIONS_%s_%s.pkl' % ( 
                                 data_path.replace('./', '').replace('/', '_'),
                                 t_start.strftime("%Y-%m-%d_%H-%M"),
                                ), )
output_filepath

WORD_FREQ_MIN=5
IMG_WORD_FREQ_MIN=5

import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords

img_to_captions=dict()

tarfilepath = os.path.join(data_path, 'flickr30k.tar.gz')
if os.path.isfile(tarfilepath):
    import tarfile
    with tarfile.open(tarfilepath, 'r:gz').extractfile('results_20130124.token') as tokenized:
        n_captions = 0
        for l in tokenized.readlines():
            #print(l)  # This is bytes
            img_num, caption = l.decode("utf-8").strip().split("\t")
            img, num = img_num.split("#")
            #print(img, caption); break
            if img not in img_to_captions:  img_to_captions[img]=[]
            img_to_captions[img].append(caption)
            n_captions += 1
            
print("Found %d images, with a total of %d captions" % (len(img_to_captions),n_captions, ))
# Found 31783 images, with a total of 158915 captions

good_img_to_captions, good_img_to_captions_title = img_to_captions, 'all'
len(good_img_to_captions)

# Filter for the images that we care about
if False:  
    # This is a super-small list, which means we won't get the chance to see 
    #   enough Text to figure out how to make sentences.  ABANDON THIS 'SIMPLIFICATION'
    import re
    good_caption = re.compile( r'\b(cat|kitten)s?\b', flags=re.IGNORECASE )
    good_img_to_captions = { img:captions
                                for img, captions in img_to_captions.items() 
                                for caption in captions 
                                if good_caption.search( caption )
                           }  # img=='3947306345.jpg'
    good_img_to_captions_title = 'feline'
    #good_img_to_captions
    len(good_img_to_captions)

img_arr = sorted(good_img_to_captions.keys())

# extract the vocab where each word is required to occur WORD_FREQ_MIN times overall
word_freq_all=dict()

#for img in img_to_captions.keys():  # everything
for img in img_arr:  # Our selection
    for caption in img_to_captions[img]:
        for w in caption.lower().split():
            if not w in word_freq_all: word_freq_all[w]=0
            word_freq_all[w] += 1
            
word_freq = { w:f for w,f in word_freq_all.items() if f>=WORD_FREQ_MIN }

freq_word = sorted([ (f,w) for w,f in word_freq.items() ], reverse=True)
vocab = set( word_freq.keys() )

len(vocab), freq_word[0:20]
# 7734,  [(271698, 'a'), (151039, '.'), (83466, 'in'), (62978, 'the'), (45669, 'on'), (44263, 'and'), ...

# extract the vocab where each word is required to occur in IMG_WORD_FREQ_MIN *images* overall
word_freq_imgs=dict()

#for img in img_to_captions.keys():  # everything
for img in img_arr:  # Our selection
    img_caption_words=set()
    for caption in img_to_captions[img]:
        for w in caption.lower().split():
            img_caption_words.add(w)
    for w in img_caption_words:
        if not w in word_freq_imgs: word_freq_imgs[w]=0
        word_freq_imgs[w] += 1
            
word_freq = { w:f for w,f in word_freq_imgs.items() if f>=IMG_WORD_FREQ_MIN }

freq_word = sorted([ (f,w) for w,f in word_freq.items() ], reverse=True)
vocab = set( word_freq.keys() )

len(vocab), freq_word[0:20]
# 7219,  [(31783, '.'), (31635, 'a'), (28076, 'in'), (24180, 'the'), (21235, 'is'), (21201, 'and'), ...

sorted([ (f,w) for w,f in word_freq.items() if not w.isalpha() and '-' not in w ], reverse=True)

stop_words = set ( stopwords.words('english') )
punc = set ("- . , : ; ' \" & $ % ( ) ! ? #".split())

[ (w, w in stop_words) for w in "while with of at in".split() ]

stop_words_seen = vocab.intersection( stop_words.union(punc) )

', '.join(stop_words_seen)
len(stop_words_seen), len(stop_words)

glove_dir = './data/RNN/'
glove_100k_50d = 'glove.first-100k.6B.50d.txt'
glove_100k_50d_path = os.path.join(glove_dir, glove_100k_50d)

if not os.path.isfile( glove_100k_50d_path ):
    raise RuntimeError("You need to download GloVE Embeddings "+
                       ": Use the downloader in 5-Text-Corpus-and-Embeddings.ipynb")
else:
    print("GloVE available locally")

# Due to size constraints, only use the first 100k vectors (i.e. 100k most frequently used words)
import glove
embedding_full = glove.Glove.load_stanford( glove_100k_50d_path )
embedding_full.word_vectors.shape

# Find words in word_arr that don't appear in GloVe
#word_arr = stop_words_seen  # Great : these all have embeddings
#word_arr = [ w for w,f in word_freq.items() if f>WORD_FREQ_MIN]  # This seems we're not missing much...
word_arr = vocab

missing_arr=[]
for w in word_arr:
    if not w in embedding_full.dictionary:
        missing_arr.append(w)
len(missing_arr), ', '.join( sorted(missing_arr) )

# Let's filter out the captions for the words that appear in our GloVe embedding
#  And ignore the images that then have no captions
img_to_valid_captions, words_used = dict(), set()
captions_total, captions_valid_total = 0,0

for img, captions in good_img_to_captions.items():
    captions_total += len(captions)
    captions_valid=[]
    for caption in captions:
        c = caption.lower()
        caption_valid=True
        for w in c.split():
            if w not in embedding_full.dictionary:
                caption_valid=False
            if w not in vocab:
                caption_valid=False
        if caption_valid:
            captions_valid.append( c )
            words_used.update( c.split() )
            
    if len(captions_valid)>0:
        img_to_valid_captions[img]=captions_valid
        captions_valid_total += len(captions_valid)
    else:
        #print("Throwing out %s" % (img,), captions)
        pass
    
print("%d images remain of %d.  %d captions remain of %d. Words used : %d" % (
            len(img_to_valid_captions.keys()), len(good_img_to_captions.keys()), 
            captions_valid_total, captions_total, 
            len(words_used),)
     )
# 31640 images remain of 31783.  135115 captions remain of 158915. Words used : 7399 (5 min appearances overall)
# 31522 images remain of 31783.  133106 captions remain of 158915. Words used : 6941 (5 min images)

# So, we only got rid of ~150 images, but 23k captions... if we require 5 mentions minimum
# And only got rid of ~250 images, but 25k captions... if we require 5 minimum image appearances

# Construct an ordered word list:
action_words = "{MASK} {UNK} {START} {STOP} {EXTRA}".split(' ')

# Then want the 'real words' to have :
#  all the stop_words_seen (so that these can be identified separately)
#  followed by the remainder of the words_used, in frequency order

def words_in_freq_order(word_arr, word_freq=word_freq):
    # Create list of freq, word pairs
    word_arr_freq = [ (word_freq[w], w) for w in word_arr]
    return [ w for f,w in sorted(word_arr_freq, reverse=True) ]

stop_words_sorted = words_in_freq_order( stop_words_seen )
rarer_words_sorted = words_in_freq_order( words_used - stop_words_seen )

#", ".join( stop_words_sorted )
#", ".join( words_in_freq_order( words_used )[0:100] ) 
#", ".join( rarer_words_sorted[0:100] ) 
len(words_used), len(action_words), len(stop_words_sorted), len(rarer_words_sorted)

EMBEDDING_DIM = embedding_full.word_vectors.shape[1]

action_embeddings = np.zeros( (len(action_words), EMBEDDING_DIM,), dtype='float32')
for idx,w  in enumerate(action_words):
    if idx>0:  # Ignore {MASK}
        action_embeddings[idx, idx] = 1.0  # Make each row a very simple (but distinct) vector for simplicity

stop_words_idx  = [ embedding_full.dictionary[w] for w in stop_words_sorted ]
rarer_words_idx = [ embedding_full.dictionary[w] for w in rarer_words_sorted ]

embedding = np.vstack([ 
        action_embeddings,
        embedding_full.word_vectors[ stop_words_idx ],
        embedding_full.word_vectors[ rarer_words_idx ],
    ])

embedding_word_arr = action_words + stop_words_sorted + rarer_words_sorted
#stop_words_idx

embedding_dictionary = { w:i for i,w in enumerate(embedding_word_arr) }

# Check that this all ties together...
#word_check='{START}'  # an action word - not found in GloVe
#word_check='this'     # a stop word
word_check='hammer'   # a 'rare' word

#embedding_dictionary[word_check]
(  embedding[ embedding_dictionary[word_check] ] [0:6], 
   embedding_full.word_vectors[ embedding_full.dictionary.get( word_check, 0) ] [0:6], )

np.random.seed(1)  # Consistent values for train/test (for this )
save_me = dict(
    img_to_captions = img_to_valid_captions,
    
    action_words = action_words, 
    stop_words = stop_words_sorted,
    
    embedding = embedding,
    embedding_word_arr = embedding_word_arr,
    
    img_arr = img_arr_save,
    train_test = np.random.random( (len(img_arr_save),) ),
)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

with open( output_filepath, 'wb') as f:
    pickle.dump(save_me, f)
    
print("Corpus saved to '%s'" % (output_filepath,))



