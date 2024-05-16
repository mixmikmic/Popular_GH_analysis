import requests
import pandas as pd
import numpy as np
import json
import time
from selenium import webdriver

pd.set_option('display.max_colwidth', 200)

browser = webdriver.PhantomJS()
browser.set_window_size(1080,800)
browser.get("http://www.ruzzit.com/en-US/Timeline?media=Articles&timeline=Year1&networks=All")
time.sleep(3)

pg_scroll_count = 50

while pg_scroll_count:
    browser.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(15)
    pg_scroll_count -= 1

titles = browser.find_elements_by_class_name("article_title")
link_class = browser.find_elements_by_class_name("link_read_more_article")
stats = browser.find_elements_by_class_name("ruzzit_statistics_area")

all_data = []
for title, link, stat in zip(titles, link_class, stats):
    all_data.append((title.text,                     link.get_attribute("href"),                     stat.find_element_by_class_name("col-md-12").text.split(' shares')[0],
                     stat.find_element_by_class_name("col-md-12").text.split('tweets\n')[1].split('likes\n0')[0],
                     stat.find_element_by_class_name("col-md-12").text.split('1\'s\n')[1].split(' pins')[0],
                     stat.find_element_by_class_name("col-md-12").text.split('pins\n')[1]))

all_data

df = pd.DataFrame(all_data, columns=['title', 'link', 'fb', 'lnkdn', 'pins', 'date'])
df

#browser.save_screenshot('/Users/alexcombs/Desktop/testimg.png')

df = df.assign(redirect = df['link'].map(lambda x: requests.get(x).url))

df

def check_home(x):
    if '.com' in x:
        if len(x.split('.com')[1]) < 2:
            return 1
        else:
            return 0
    else:
        return 0

def check_img(x):
    if '.gif' in x or '.jpg' in x:
        return 1
    else:
        return 0

df = df.assign(pg_missing = df['pg_missing'].map(check_home))

df = df.assign(img_link = df['redirect'].map(check_img))

df

df[df['pg_missing']==1]

len(df[df['pg_missing']==1])

len(df[df['img_link']==1])

df[df['pg_missing']==1]

dfc = df[(df['img_link']!=1)&(df['pg_missing']!=1)]

dfc

def get_data(x):
    try:
        data = requests.get('https://api.embedly.com/1/extract?key=SECRET_KEY&url=' + x)
        json_data = json.loads(data.text)
        return json_data
    except:
        print('Failed')
        return None

dfc = dfc.assign(json_data = dfc['redirect'].map(get_data))

dfc_bak = dfc

dfc

def get_title(x):
    try:
        return x.get('title')
    except:
        return None

dfc = dfc.assign(title = dfc['json_data'].map(get_title))

def get_site(x):
    try:
        return x.get('provider_name')
    except:
        return None

dfc = dfc.assign(site = dfc['json_data'].map(get_site))

def get_images(x):
    try:
        return len(x.get('images'))
    except:
        return None

dfc = dfc.assign(img_count = dfc['json_data'].map(get_images))

def get_entities(x):
    try:
        return [y.get('name') for y in x.get('entities')]
    except:
        return None

dfc = dfc.assign(entities = dfc['json_data'].map(get_entities))

def get_html(x):
    try:
        return x.get('content')
    except:
        return None

dfc = dfc.assign(html = dfc['json_data'].map(get_html))

dfc[::-1]

from bs4 import BeautifulSoup

def text_from_html(x):
    try:
        soup = BeautifulSoup(x, 'lxml')
        return soup.get_text()
    except:
        return None

dfc = dfc.assign(text = dfc['html'].map(text_from_html))

dfc[::-1]

def clean_counts(x):
    if 'M' in str(x):
        d = x.split('M')[0]
        dm = float(d) * 1000000
        return dm
    elif 'k' in str(x):
        d = x.split('k')[0]
        dk = float(d.replace(',','')) * 1000
        return dk
    elif ',' in str(x):
        d = x.replace(',','')
        return int(d)
    else:
        return x

dfc = dfc.assign(fb = dfc['fb'].map(clean_counts))

dfc = dfc.assign(lnkdn = dfc['lnkdn'].map(clean_counts))

dfc = dfc.assign(pins = dfc['pins'].map(clean_counts))

dfc = dfc.assign(date = pd.to_datetime(dfc['date'], dayfirst=True))

dfc

def get_word_count(x):
    if not x is None:
        return len(x.split(' '))
    else:
        return None

dfc = dfc.assign(word_count = dfc['text'].map(get_word_count))

dfc[['text','word_count']][::-1]

import matplotlib.colors as mpc

def get_hex(x):
    try:
        if x.get('images'):
            main_color = x.get('images')[0].get('colors')[0].get('color')
            return mpc.rgb2hex([(x/255) for x in main_color])
    except:
        return None

def get_rgb(x):
    try:
        if x.get('images'):
            main_color = x.get('images')[0].get('colors')[0].get('color')
            return main_color
    except:
        return None

dfc = dfc.assign(main_hex = dfc['json_data'].map(get_hex))
dfc = dfc.assign(main_rgb = dfc['json_data'].map(get_rgb))

dfc

dfc['img_count'].value_counts().to_frame('count')

fig, ax = plt.subplots(figsize=(8,6))
y = dfc['img_count'].value_counts().sort_index()
x = y.sort_index().index
plt.bar(x, y, color='k', align='center')
plt.title('Image Count Frequency', fontsize=16, y=1.01)
ax.set_xlim(-.5,5.5)
ax.set_ylabel('Count')
ax.set_xlabel('Number of Images')

#dfc.to_json('/Users/alexcombs/Desktop/viral_data.json')
dfc = pd.read_json('/Users/alexcombs/Desktop/viral_data.json')

mci = dfc['main_hex'].value_counts().to_frame('count')
mci

mci['color'] = ' '

def color_cells(x):
    return 'background-color: ' + x.index

mci.style.apply(color_cells, subset=['color'], axis=0)

def get_csplit(x):
    try:
        return x[0], x[1], x[2]
    except:
        return None, None, None

dfc['reds'], dfc['greens'], dfc['blues'] = zip(*dfc['main_rgb'].map(get_csplit))

dfc

from sklearn.cluster import KMeans

np.sqrt(256)

clf = KMeans(n_clusters=16)

clf.fit(dfc[['reds', 'greens', 'blues']].dropna())

clusters = pd.DataFrame(clf.cluster_centers_, columns=['r', 'g', 'b'])

clusters

def hexify(x):
    rgb = [round(x['r']), round(x['g']), round(x['b'])]
    hxc = mpc.rgb2hex([(x/255) for x in rgb])
    return hxc

clusters.index = clusters.apply(hexify, axis=1)

clusters

clusters['color'] = ' '

clusters

clusters.style.apply(color_cells, subset=['color'], axis=0)

dfc[dfc['title'].isnull()]

from nltk.util import ngrams
from nltk.corpus import stopwords
import re

def get_word_stats(txt_series, n, rem_stops=False):
    txt_words = []
    txt_len = []
    for w in txt_series:
        if w is not None:
            if rem_stops == False:
                word_list = [x for x in ngrams(re.findall('[a-z0-9\']+', w.lower()), n)]
            else:
                word_list = [y for y in ngrams([x for x in re.findall('[a-z0-9\']+', w.lower())                                                if x not in stopwords.words('english')], n)]
            word_list_len = len(list(word_list))
            txt_words.extend(word_list)
            txt_len.append(word_list_len)
    return pd.Series(txt_words).value_counts().to_frame('count'), pd.DataFrame(txt_len, columns=['count'])

hw,hl = get_word_stats(dfc['title'], 3, 1)

hw

hl.describe()

tt = dfc[~dfc['title'].isnull()]

tt[tt['title'].str.contains('Dies')]

dfc['site'].value_counts().to_frame()

hw,hl = get_word_stats(dfc['text'], 3, 1)

hw

from sklearn.ensemble import RandomForestRegressor

all_data = dfc.dropna(subset=['img_count', 'word_count'])

all_data.reset_index(inplace=True, drop=True)

all_data

train_index = []
test_index = []
for i in all_data.index:
    result = np.random.choice(2, p=[.65,.35])
    if result == 1:
        test_index.append(i)
    else:
        train_index.append(i)

print('test length:', len(test_index), '\ntrain length:', len(train_index))

sites = pd.get_dummies(all_data['site'])

sites

y_train = all_data.iloc[train_index]['fb'].astype(int)
X_train_nosite = all_data.iloc[train_index][['img_count', 'word_count']]

X_train = pd.merge(X_train_nosite, sites.iloc[train_index], left_index=True, right_index=True)

y_test = all_data.iloc[test_index]['fb'].astype(int)
X_test_nosite = all_data.iloc[test_index][['img_count', 'word_count']]

X_test = pd.merge(X_test_nosite, sites.iloc[test_index], left_index=True, right_index=True)

clf = RandomForestRegressor(n_estimators=100)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

y_actual = y_test

deltas = pd.DataFrame(list(zip(y_pred, y_actual, (y_pred - y_actual)/(y_actual))), columns=['predicted', 'actual', 'delta'])

deltas

deltas['delta'].describe()

a = pd.Series([10,10,10,10])
b = pd.Series([12,8,8,12])

np.sqrt(np.mean((b-a)**2))/np.mean(a)

(b-a).mean()

np.sqrt(np.mean((y_pred-y_actual)**2))/np.mean(y_actual)



from sklearn.feature_extraction.text import CountVectorizer

vect = CountVectorizer(ngram_range=(1,3))

X_titles_all = vect.fit_transform(all_data['title'])

X_titles_train = X_titles_all[train_index]

X_titles_test = X_titles_all[test_index]

len(X_titles_train.toarray())

len(X_titles_test.toarray())

len(X_train)

len(X_test)

X_test = pd.merge(X_test, pd.DataFrame(X_titles_test.toarray(), index=X_test.index), left_index=True, right_index=True)

X_train = pd.merge(X_train, pd.DataFrame(X_titles_train.toarray(), index=X_train.index), left_index=True, right_index=True)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

deltas = pd.DataFrame(list(zip(y_pred, y_actual, (y_pred - y_actual)/(y_actual))), columns=['predicted', 'actual', 'delta'])

deltas

np.sqrt(np.mean((y_pred-y_actual)**2))/np.mean(y_actual)

all_data = all_data.assign(title_wc = all_data['title'].map(lambda x: len(x.split(' '))))

X_train = pd.merge(X_train, all_data[['title_wc']], left_index=True, right_index=True)

X_test = pd.merge(X_test, all_data[['title_wc']], left_index=True, right_index=True)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

y_actual = y_test

np.sqrt(np.mean((y_pred-y_actual)**2))/np.mean(y_actual)





