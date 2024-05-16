from imp import reload
import numpy as np
import menu, preprocess, models
reload(menu)

menuitems = [('1', 'choice 1', lambda: 'you chose 1'),
             ('2', 'choice 2', lambda: 'you chose 2')
            ]

m = menu.Menu('00', 'main menu', menuitems)

x = menu.Choice(menuitems[0])

res = m()

res

'~' * 3

input('enter')

reload(preprocess)
reload(models)
ve = preprocess.BabiVectorizer()
ve.vectorize_query('Where is John?', verbose=True)

def charvectorize(word, lower=True):
    if lower:
        word = word.lower()
    idxs = [ord(c) for c in word]
    vec = np.zeros(128, int)
    for c in word:
        vec[ord(c)] = 1
    return vec
    
def dist(v1, v2):
    dv = v2 - v1
    dv = dv**2
    dv = np.sum(dv, axis=-1)
    return dv**0.5

def softdist(word1, word2, lower=True):
    v1 = charvectorize(word1, lower)
    v2 = charvectorize(word2, lower)
    return dist(v1, v2)
    
    
def matchnocase(word, vocab):
    lword = word.lower()
    listvocab = list(vocab)
    lvocab = [w.lower() for w in listvocab]
    if lword in lvocab:
        return listvocab[lvocab.index(lword)]
    return None
    

def softmatch(word, vocab, cutoff=2.):
    """Try to soft-match to catch various typos. """
    vw = charvectorize(word)
    vecs = np.array([charvectorize(w) for w in vocab])
    print(vecs.shape)
    distances = dist(vw, vecs)
    idx = np.argmin(distances)
    confidence = distances[idx]
    if confidence < cutoff:
        return vocab[idx]
    return None
    
softmatch('john?', list(ve.word_idx))
# matchnocase('MAry', ve.word_idx)

import os

os.path.normpath()

os.sep

ll

fname = 'foo/bar//spam.txt'
os.makedirs(os.path.dirname(fname), exist_ok=True)

os.path.normpath(fname)



