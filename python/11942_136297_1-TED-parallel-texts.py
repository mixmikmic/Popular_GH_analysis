import requests
from bs4 import BeautifulSoup
#import shutil
#import codecs
import os, glob
import csv
import time, random

def enlist_talk_names(url, dict_):
    time.sleep( random.random()*5.0+5.0 )
    r = requests.get(url)
    print("  Got %d bytes from %s" % (len(r.text), url))
    soup = BeautifulSoup(r.text, 'html.parser')
    talks= soup.find_all("a", class_='')
    for i in talks:
        if i.attrs['href'].find('/talks/')==0 and dict_.get(i.attrs['href'])!=1:
            dict_[i.attrs['href']]=1
    return dict_

all_talk_names={}

# Get all pages of talks (seems a bit abusive)
#for i in xrange(1,61):
#    url='https://www.ted.com/talks?page=%d'%(i)
#    all_talk_names=enlist_talk_names(url, all_talk_names)

# A specific seach term
#url='https://www.ted.com/talks?sort=newest&q=ai'

# Specific topics
url='https://www.ted.com/talks?sort=newest&topics[]=AI'
#url='https://www.ted.com/talks?sort=newest&topics[]=machine+learning'
#url='https://www.ted.com/talks?sort=newest&topics[]=mind'
#url='https://www.ted.com/talks?sort=newest&topics[]=mind&page=2'
all_talk_names=enlist_talk_names(url, all_talk_names)
len(all_talk_names)

data_path = './data'
if not os.path.exists(data_path):
    os.makedirs(data_path)

def extract_talk_languages(url, talk_name, language_list=['en', 'ko', 'ja']):
    need_more_data=False
    for lang in language_list:
        talk_lang_file = os.path.join(data_path, talk_name+'-'+lang+'.csv')
        if not os.path.isfile( talk_lang_file ) :
            need_more_data=True
    if not need_more_data:
        print("  Data already retrieved for %s" % (url,))
        return

    time.sleep( random.random()*5.0+5.0 )
    r = requests.get(url)
    print("  Got %d bytes from %s" % (len(r.text), url))
    if len(r.text)<1000: return # FAIL!
    soup = BeautifulSoup(r.text, 'html.parser')
    for i in soup.findAll('link'):
        if i.get('href')!=None and i.attrs['href'].find('?language=')!=-1:
            #print i.attrs['href']
            lang=i.attrs['hreflang']
            url_lang=i.attrs['href']
            if not lang in language_list:
                continue
                
            talk_lang_file = os.path.join(data_path, talk_name+'-'+lang+'.csv')
            if os.path.isfile( talk_lang_file ) :
                continue
                
            time.sleep( random.random()*5.0+5.0 )
            r_lang = requests.get(url_lang)
            print("    Lang[%s] : Got %d bytes" % (lang, len(r_lang.text), ))
            if len(r.text)<1000: return # FAIL!
            lang_soup = BeautifulSoup(r_lang.text, 'html.parser')

            talk_data = []
            for i in lang_soup.findAll('span',class_='talk-transcript__fragment'):
                d = [ int( i.attrs['data-time'] ), i.text.replace('\n',' ') ]
                talk_data.append(d)
            
            with open(talk_lang_file, 'w') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['ts', 'txt'])
                writer.writerows(talk_data)            

if False:
    # Now flatten out the talk_data into time_step order
    talk_data_csv = [ ['ts']+language_list, ]
    for ts in sorted(talk_data.keys(), key=int):
        row = [ts] + [ talk_data[ts].get(lang, '') for lang in language_list]
        talk_data_csv.append(row)
        
    with open(os.path.join(data_path, talk_name+'.csv'), 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(talk_data_csv)

for name in all_talk_names:
    extract_talk_languages('https://www.ted.com'+name+'/transcript', name[7:])
    #break
print("Finished extract_talk_languages for all_talk_names")



