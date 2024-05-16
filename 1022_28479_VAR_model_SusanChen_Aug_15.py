import pandas as pd
import numpy as np
import statsmodels.tsa.api as tsa
import pandas.io.data as web
import datetime

#FRED Remote Data Access API for Real GDP data 

start = datetime.datetime(1995, 1, 1)
end = datetime.datetime(2014, 12, 30)

gdp=web.DataReader(['NAEXKP01FRQ189S','NAEXKP01ITQ189S','NAEXKP01DKQ189S','NAEXKP01SEQ652S','NAEXKP01ESQ652S','NAEXKP01PTQ652S','NAEXKP01DEQ189S','NAEXKP01NLQ189S','NAEXKP01BEQ189S','NAEXKP01ATQ189S','NAEXKP01FIQ189S','NAEXKP01GBQ652S'], 
"fred", start, end)

gdp.columns = ['France', 'Italy', 'Denmark', 'Sweden', 'Spain', 'Portugal', 'Germany', 'Netherlands', 'Belgium', 'Austria', 'Finland', 'United Kingdom']

#FRED Remote Data Access API for Exchange rate data 

exchange=web.DataReader(['CCUSMA02FRQ618N','CCUSMA02ITQ618N','CCUSMA02DKQ618N','CCUSMA02SEQ618N','CCUSMA02ESQ618N','CCUSMA02PTQ618N','CCUSMA02DEQ618N','CCUSMA02NLQ618N' ,'CCUSMA02BEQ618N', 'CCUSMA02ATQ618N','CCUSMA02FIQ618N','CCUSMA02GBQ618N'],
"fred", start, end)

exchange.columns = ['France', 'Italy', 'Denmark', 'Sweden', 'Spain', 'Portugal', 'Germany', 'Netherlands', 'Belgium', 'Austria', 'Finland', 'United Kingdom']

#Data downloaded from OECD and read into python using pandas

file1 = '/users/susan/desktop/cpiquarterlyoecd.csv' # file location #dates row replaced with a datetime format
cpi_df = pd.read_csv(file1) 
cpi_df = cpi_df.transpose() #OECD has years as columns and countries as rows 
cpi = cpi_df.drop(cpi_df.index[0]) #drop blank 'location' row
cpi.index = pd.to_datetime(cpi.index) #convert dates to datetime format
cpi.columns = ['Austria','Belgium','Czech Republic','Denmark','Estonia','Finland','France','Germany','Greece','Hungary','Ireland','Italy','Latvia','Luxembourg','Netherlands','Poland','Portugal','Slovak Republic','Slovenia','Spain','Sweden','United Kingdom']

file2 = '/users/susan/desktop/interestquarterlyoecd.csv' # file location
interest_df = pd.read_csv(file2)
interest_df = interest_df.transpose() #OECD has years as columns and countries as rows 
interest = interest_df.drop(interest_df.index[0]) #drop blank 'location' row
interest.index = pd.to_datetime(interest.index) #convert dates to datetime format
interest.columns = ['Austria','Belgium','Czech Republic','Denmark','Estonia','Finland','France','Germany','Greece','Hungary','Ireland','Italy','Latvia','Luxembourg','Netherlands','Poland','Portugal','Slovak Republic','Slovenia','Spain','Sweden','United Kingdom']

#Creating a list of dataframes organized by country 

by_country = {}

for country in gdp.columns:
    country_df = pd.concat([gdp[country], cpi[country], interest[country]], axis = 1) #add exchange[country] if including exchange rates
    country_df.columns = ['RealGDP', 'CPI', 'Interest'] #add 'Exchange' if necessary 
    country_df = country_df.convert_objects(convert_numeric = True)
    country_df = country_df.dropna()
    by_country[country] = country_df

def fuller(country):
    for country in gdp.columns:
        print (country)    
        print ('Real GDP:' , (tsa.stattools.adfuller(by_country[country].RealGDP, maxlag = 2, regression = 'ct'))[0]) #prints the t-statistic
        print ('CPI:' , tsa.stattools.adfuller(by_country[country].CPI, maxlag = 2, regression = 'ct')[0])
        print ('Interest:' , tsa.stattools.adfuller(by_country[country].Interest, maxlag = 2, regression = 'ct')[0])
        print ('---')

fuller(country)

def varmodel(country):
    mdata = by_country[country]
    data = np.log(mdata)
    model = tsa.VAR(data)
    res = model.fit(model.select_order()['bic'])
    irf = res.irf(24)
    irf.plot()
    #print (res.summary())  
    #print (res.is_stable())

varmodel('France')

varmodel('Italy')

varmodel('Denmark')

varmodel('Sweden')

varmodel('Spain')

varmodel('Portugal')

varmodel('Germany')

varmodel('Netherlands')

varmodel('Belgium')

varmodel('Austria')

varmodel('Finland')

varmodel('United Kingdom')

