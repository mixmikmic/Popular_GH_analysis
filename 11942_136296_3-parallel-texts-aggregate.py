import os
import csv
import time, random
import re

lang_from, lang_to = 'en', 'ko'

data_path = './data'

stub_from, stub_to = set(),set()
stub_matcher = re.compile(r"(.*)\-(\w+)\.csv")
for fname in os.listdir(data_path):
    #print(fname)
    m = stub_matcher.match(fname)
    if m:
        stub, lang = m.group(1), m.group(2)
        if lang == lang_from: stub_from.add(stub)
        if lang == lang_to:   stub_to.add(stub)
stub_both = stub_from.intersection(stub_to)

correspondence_loc,txt_from,txt_to=[],[],[]

def read_dict_from_csv(fname):
    d=dict()
    with open(fname, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            d[float(row['ts'])]=row['txt']
    return d

for stub in stub_both:
    #print("Reading stub %s" % (stub,))
    data_from = read_dict_from_csv( os.path.join(data_path, stub+'-'+lang_from+'.csv') )
    data_to   = read_dict_from_csv( os.path.join(data_path, stub+'-'+lang_to+'.csv') )
    
    valid, skipped=0, 0
    for ts, txt in data_from.items():
        if ts in data_to:
            correspondence_loc.append( (stub, ts) )
            txt_from.append( txt )
            txt_to.append( data_to[ts] )
            valid += 1
        else:
            skipped += 1
    print("%3d valid of %3d fragments from '%s'" % (valid, valid+skipped, stub))
print("  Total data : %d text fragments" % (len(correspondence_loc),)) 

for _ in range(10):
    i = random.randrange(len(correspondence_loc))
    print( txt_from[i], txt_to[i]  )

sub_punctuation = re.compile(r'[\,\.\:\;\?\!\-\—\s\"0-9\(\)]+')
sub_apostrophes = re.compile(r'\'(\w+)')
sub_multispaces = re.compile(r'\s\s+')
    
if lang_from=='ja' or lang_to=='ja':
    import tinysegmenter
    ja_segmenter = tinysegmenter.TinySegmenter()
    sub_punc_ja  = re.compile(r'[\」\「\？\。\、\・\（\）\―]+')

def tokenize_txt(arr, lang):
    tok=[]
    for txt in arr:
        t = txt.lower()
        t = re.sub(sub_punctuation, u' ', t)
        if "'" in t:
            t = re.sub(sub_apostrophes, r" '\1", t)
        if lang=='ja':
            t = ' '.join( ja_segmenter.tokenize(t) )
            t = re.sub(sub_punc_ja, u' ', t)
        t = re.sub(sub_multispaces, ' ', t)
        tok.append(t.strip())
    return tok

tok_from = tokenize_txt(txt_from, lang_from)
tok_to   = tokenize_txt(txt_to, lang_to)

tok_from[220:250]

tok_to[220:250]

def build_freq(tok_arr):
    f=dict()
    for tok in tok_arr:
        for w in tok.split():
            if w not in f: f[w]=0
            f[w]+=1
    return f

freq_from=build_freq(tok_from)
freq_to  =build_freq(tok_to)

len(freq_from),len(freq_to), 

def most_frequent(freq, n=50, start=0):
    return ', '.join( sorted(freq,key=lambda w:freq[w], reverse=True)[start:n+start] )

print(most_frequent(freq_from))
print(most_frequent(freq_to, n=100))

print(most_frequent(freq_from, n=20, start=9000))

print( len( [_ for w,f in freq_from.items() if f>=10]))
print( len( [_ for w,f in freq_to.items() if f>=10]))

def build_rank(freq):
    return { w:i for i,w in enumerate( sorted(freq, key=lambda w:freq[w], reverse=True) ) }

rank_from = build_rank(freq_from)
rank_to   = build_rank(freq_to)

print(rank_from['robot'])

def max_rank(tok, rank):  # Find the most infrequent word in this tokenized sentence
    r = -1
    for w in tok.split():
        if rank[w]>r: r=rank[w] 
    return r
tok_max_rank_from = [ max_rank(tok, rank_from) for tok in tok_from ]
tok_max_rank_to   = [ max_rank(tok, rank_to)   for tok in tok_to ]

start=0;print(tok_max_rank_from[start:start+15], '\n', tok_max_rank_to[start:start+15],)
i=0; tok_max_rank_from[i], tok_from[i], tok_to[i], tok_max_rank_to[i], 



