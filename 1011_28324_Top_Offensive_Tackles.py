from bs4 import BeautifulSoup
import urllib2
import pandas as pd
from pandas import DataFrame, Series
get_ipython().magic('matplotlib inline')
from __future__ import division
from matplotlib import pyplot as plt
import seaborn as sns
sns.set(style='ticks', palette='Set2')
import statsmodels.api as sm

html = urllib2.urlopen('http://espn.go.com/college-sports/football/recruiting/playerrankings/_/view/position/order/true/position/offensive-guard')
text = html.read()
soup = BeautifulSoup(text.replace('ISO-8859-1', 'utf-8'))

ht_wgt = []
for tr in soup.findAll('tr')[1:]:
    tds = tr.findAll('td')
    height = tds[4].text
    weight = tds[5].text
    grade = tds[7].text
    ht_wgt.append([height, weight, grade])

#should have 100
len(ht_wgt)

data = DataFrame(ht_wgt, columns=['height', 'weight', 'grade'])
data.head()

data['weight'] = data.weight.astype(int)
data['grade'] = data.grade.astype(int)
hgt_str = data.height.values
hgt_str = [x.split("'") for x in hgt_str]
hgt_in = [(int(x[0]) * 12) + int(x[1]) for x in hgt_str]
data['height_inches'] = hgt_in
data['grade_meanzero'] = data.grade - data.grade.mean()
data.head()

fig, ax = plt.subplots(3,1)
fig.set_size_inches(8.5, 11)

fig.suptitle('2015 ESPN Top 100 High School Offensive tackles',
             fontsize=16, fontweight='bold')

ax[0].hist(data.height_inches, bins = range(73,83), align='left')
ax[0].set_xlabel('Height')
ax[0].set_ylabel('Number of Players')
ax[0].annotate('Average Height: {}'.format(data.height_inches.mean()), 
             xy=(.5, .5), xytext=(.70, .7),  
             xycoords='axes fraction', textcoords='axes fraction')
ax[0].plot([75, 75], [0,40])
ax[0].set_xlim([72,82])
ax[0].set_xticks(range(73,83))
ax[0].annotate('My Brother', xy=(75, 20), xytext=(73, 25))

ax[1].hist(data.weight)
ax[1].set_xlabel('Weight in Pounds')
ax[1].set_ylabel('Number of Players')
ax[1].annotate('Average Weight: {}'.format(data.weight.mean()), 
             xy=(.5, .5), xytext=(.70, .7),  
             xycoords='axes fraction', textcoords='axes fraction')
ax[1].plot([280, 280], [0,30])
ax[1].annotate('My Brother', xy=(250, 15), xytext=(236, 20))

ax[2].scatter(data.height_inches, data.weight, s=data.grade_meanzero*15, alpha=.6)
ax[2].set_title('Bigger Circle Means Better Rank')
ax[2].set_xlabel('Height in Inches')
ax[2].set_ylabel('Weight in Pounds')
ax[2].set_xlim([72,82])
ax[2].set_xticks(range(73,83))
ax[2].scatter([75],[280], alpha=1, s=50, c=sns.color_palette("Set2", 2)[1])
ax[2].annotate('My Brother', xy=(75, 280), xytext=(73.5, 255))

fig.tight_layout()
plt.subplots_adjust(top=0.92)
sns.despine()
plt.savefig('Top100_OT.png')

