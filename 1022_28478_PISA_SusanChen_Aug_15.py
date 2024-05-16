get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from pandas.io import wb

file1 = '/users/susan/desktop/PISA/PISA2012clean.csv' # file location
df1 = pd.read_csv(file1)

#pandas remote data access API for World Bank GDP per capita data
df2 = wb.download(indicator='NY.GDP.PCAP.PP.KD', country='all', start=2012, end=2012)

df1

#drop multilevel index 
df2.index = df2.index.droplevel('year') 

df1.columns = ['Country','Math','Reading','Science']
df2.columns = ['GDPpc']

#combine PISA and GDP datasets based on country column  
df3 = pd.merge(df1, df2, how='left', left_on = 'Country', right_index = True) 

df3.columns = ['Country','Math','Reading','Science','GDPpc']

#drop rows with missing GDP per capita values
df3 = df3[pd.notnull(df3['GDPpc'])] 

print (df3)

df3.index = df3.Country #set country column as the index 
df3 = df3.drop(['Qatar', 'Vietnam']) # drop outlier

Reading = df3.Reading
Science = df3.Science
Math = df3.Math
GDP = np.log(df3.GDPpc)

#PISA reading vs GDP per capita
plt.scatter(x = GDP, y = Reading, color = 'r') 
plt.title('PISA 2012 Reading scores vs. GDP per capita')
plt.xlabel('GDP per capita (log)')
plt.ylabel('PISA Reading Score')
plt.show()

#PISA math vs GDP per capita
plt.scatter(x = GDP, y = Math, color = 'b')
plt.title('PISA 2012 Math scores vs. GDP per capita')
plt.xlabel('GDP per capita (log)')
plt.ylabel('PISA Math Score')
plt.show()

#PISA science vs GDP per capita
plt.scatter(x = GDP, y = Science, color = 'g')
plt.title('PISA 2012 Science scores vs. GDP per capita')
plt.xlabel('GDP per capita (log)')
plt.ylabel('PISA Science Score')
plt.show()

lm = smf.ols(formula='Reading ~ GDP', data=df3).fit()
lm2.params
lm.summary()

lm2 = smf.ols(formula='Math ~ GDP', data=df3).fit()
lm2.params
lm2.summary()

lm3 = smf.ols(formula='Science ~ GDP', data=df3).fit()
lm3.params
lm3.summary()

