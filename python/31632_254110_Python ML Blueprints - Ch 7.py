import pandas as pd
import numpy as np
from pandas_datareader import data, wb
import matplotlib.pyplot as plt

get_ipython().magic('matplotlib inline')
pd.set_option('display.max_colwidth', 200)

#!pip install pandas_datareader

import pandas_datareader as pdr

start_date = pd.to_datetime('2010-01-01')
stop_date = pd.to_datetime('2016-03-01')

spy = pdr.data.get_data_yahoo('SPY', start_date, stop_date)

spy

spy_c = spy['Close']

fig, ax = plt.subplots(figsize=(15,10))
spy_c.plot(color='k')
plt.title("SPY", fontsize=20)

first_open = spy['Open'].iloc[0]
first_open

last_close = spy['Close'].iloc[-1]
last_close

last_close - first_open

spy['Daily Change'] = pd.Series(spy['Close'] - spy['Open'])

spy['Daily Change'].sum()

np.std(spy['Daily Change'])

spy['Overnight Change'] = pd.Series(spy['Open'] - spy['Close'].shift(1))

spy['Overnight Change'].sum()

np.std(spy['Overnight Change'])

# daily returns
daily_rtn = ((spy['Close'] - spy['Close'].shift(1))/spy['Close'].shift(1))*100
daily_rtn

daily_rtn.hist(bins=50, color='lightblue', figsize=(12,8))

# intra day returns
id_rtn = ((spy['Close'] - spy['Open'])/spy['Open'])*100
id_rtn

id_rtn.hist(bins=50, color='lightblue', figsize=(12,8))

# overnight returns
on_rtn = ((spy['Open'] - spy['Close'].shift(1))/spy['Close'].shift(1))*100
on_rtn

on_rtn.hist(bins=50, color='lightblue', figsize=(12,8))

def get_stats(s, n=252):
    s = s.dropna()
    wins = len(s[s>0])
    losses = len(s[s<0])
    evens = len(s[s==0])
    mean_w = round(s[s>0].mean(), 3)
    mean_l = round(s[s<0].mean(), 3)
    win_r = round(wins/losses, 3)
    mean_trd = round(s.mean(), 3)
    sd = round(np.std(s), 3)
    max_l = round(s.min(), 3)
    max_w = round(s.max(), 3)
    sharpe_r = round((s.mean()/np.std(s))*np.sqrt(n), 4)
    cnt = len(s)
    print('Trades:', cnt,          '\nWins:', wins,          '\nLosses:', losses,          '\nBreakeven:', evens,          '\nWin/Loss Ratio', win_r,          '\nMean Win:', mean_w,          '\nMean Loss:', mean_l,          '\nMean', mean_trd,          '\nStd Dev:', sd,          '\nMax Loss:', max_l,          '\nMax Win:', max_w,          '\nSharpe Ratio:', sharpe_r)

get_stats(daily_rtn)

get_stats(id_rtn)

get_stats(on_rtn)

def get_signal(x):
    val = np.random.rand()
    if val > .5:
        return 1
    else:
        return 0

for i in range(1000):
    spy['Signal_' + str(i)] = spy.apply(get_signal, axis=1)

spy

#spy.to_csv('/Users/alexcombs/Downloads/spy.csv', index=False)
spy = pd.read_csv('/Users/alexcombs/Downloads/spy.csv')
#spy.drop([x for x in spy.columns is 'Signal' in x])

sumd={}
for i in range(1000):
    sumd.update({i: np.where(spy['Signal_' + str(i)].iloc[1:]==1, spy['Overnight Change'].iloc[1:],0).sum()})

returns = pd.Series(sumd).to_frame('return').sort_values('return', ascending=0)

returns

mystery_rtn = pd.Series(np.where(spy['Signal_270'].iloc[1:]==1,spy['Overnight Change'].iloc[1:],0))

get_stats(mystery_rtn)

start_date = pd.to_datetime('2000-01-01')
stop_date = pd.to_datetime('2016-03-01')

sp = pdr.data.get_data_yahoo('SPY', start_date, stop_date)

sp

fig, ax = plt.subplots(figsize=(15,10))
sp['Close'].plot(color='k')
plt.title("SPY", fontsize=20)

long_day_rtn = ((sp['Close'] - sp['Close'].shift(1))/sp['Close'].shift(1))*100

(sp['Close'] - sp['Close'].shift(1)).sum()

get_stats(long_day_rtn)

long_id_rtn = ((sp['Close'] - sp['Open'])/sp['Open'])*100

(sp['Close'] - sp['Open']).sum()

get_stats(long_id_rtn)

long_on_rtn = ((sp['Open'] - sp['Close'].shift(1))/sp['Close'].shift(1))*100

(sp['Open'] - sp['Close'].shift(1)).sum()

get_stats(long_on_rtn)

for i in range(1, 21, 1):
    sp.loc[:,'Close Minus ' + str(i)] = sp['Close'].shift(i)

sp

sp20 = sp[[x for x in sp.columns if 'Close Minus' in x or x == 'Close']].iloc[20:,]

sp20

sp20 = sp20.iloc[:,::-1]

sp20

from sklearn.svm import SVR

clf = SVR(kernel='linear')

len(sp20)

X_train = sp20[:-2000]
y_train = sp20['Close'].shift(-1)[:-2000]

X_test = sp20[-2000:-1000]
y_test = sp20['Close'].shift(-1)[-2000:-1000]

model = clf.fit(X_train, y_train)

preds = model.predict(X_test)

preds

len(preds)

tf = pd.DataFrame(list(zip(y_test, preds)), columns=['Next Day Close', 'Predicted Next Close'], index=y_test.index)

tf

cdc = sp[['Close']].iloc[-2000:-1000]
ndo = sp[['Open']].iloc[-2000:-1000].shift(-1)

tf1 = pd.merge(tf, cdc, left_index=True, right_index=True)
tf2 = pd.merge(tf1, ndo, left_index=True, right_index=True)
tf2.columns = ['Next Day Close', 'Predicted Next Close', 'Current Day Close', 'Next Day Open']

tf2

def get_signal(r):
    if r['Predicted Next Close'] > r['Next Day Open'] + 1:
        return 0
    else:
        return 1

def get_ret(r):
    if r['Signal'] == 1:
        return ((r['Next Day Close'] - r['Next Day Open'])/r['Next Day Open']) * 100
    else:
        return 0

tf2 = tf2.assign(Signal = tf2.apply(get_signal, axis=1))
tf2 = tf2.assign(PnL = tf2.apply(get_ret, axis=1))

tf2

(tf2[tf2['Signal']==1]['Next Day Close'] - tf2[tf2['Signal']==1]['Next Day Open']).sum()

(sp['Close'].iloc[-2000:-1000] - sp['Open'].iloc[-2000:-1000]).sum()

get_stats((sp['Close'].iloc[-2000:-1000] - sp['Open'].iloc[-2000:-1000])/sp['Open'].iloc[-2000:-1000] * 100)

get_stats(tf2['PnL'])

#!pip install fastdtw

from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

def dtw_dist(x, y):
    distance, path = fastdtw(x, y, dist=euclidean)
    return distance

tseries = []
tlen = 5
for i in range(tlen, len(sp), tlen):
    pctc = sp['Close'].iloc[i-tlen:i].pct_change()[1:].values * 100
    res = sp['Close'].iloc[i-tlen:i+1].pct_change()[-1] * 100
    tseries.append((pctc, res))

len(tseries)

tseries[0]

dist_pairs = []
for i in range(len(tseries)):
    for j in range(len(tseries)):
        dist = dtw_dist(tseries[i][0], tseries[j][0])
        dist_pairs.append((i,j,dist,tseries[i][1], tseries[j][1]))

dist_frame = pd.DataFrame(dist_pairs, columns=['A','B','Dist', 'A Ret', 'B Ret'])

sf = dist_frame[dist_frame['Dist']>0].sort_values(['A','B']).reset_index(drop=1)

sfe = sf[sf['A']<sf['B']]

winf = sfe[(sfe['Dist']<=1)&(sfe['A Ret']>0)]

winf

plt.plot(np.arange(4), tseries[6][0])

plt.plot(np.arange(4), tseries[598][0])

excluded = {}
return_list = []
def get_returns(r):
    if excluded.get(r['A']) is None:
        return_list.append(r['B Ret'])
        if r['B Ret'] < 0:
            excluded.update({r['A']:1})

winf.apply(get_returns, axis=1);

get_stats(pd.Series(return_list))





