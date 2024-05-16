import pandas as pd   #The data package
import sys            #The code below wont work for any versions before Python 3. This just ensures that (allegedly).
assert sys.version_info >= (3,5)


import requests
import io
import zipfile     #Three packages we'll need to unzip the data

"""
The next two lines of code converts the URL into a format that works
with the "zipfile" package.
"""
url2013 = 'http://www.federalreserve.gov/econresdata/scf/files/scfp2013s.zip'
url2013_requested = requests.get(url2013)  

"""
Next, zipfile downloads, unzips, and saves the file to your computer. 'url2013_unzipped' 
contains the file path for the file.
"""
zipfile2013 = zipfile.ZipFile(io.BytesIO(url2013_requested.content))        
url2013_unzipped = zipfile2013.extract(zipfile2013.namelist()[0]) 


df2013 = pd.read_stata(url2013_unzipped)



df2013.head(10)       #Returns the first 10 rows of the dataframe

def unzip_survey_file(year = '2013'):
    import requests, io, zipfile
    import pandas as pd
    
    if int(year) <1989:
        url = 'http://www.federalreserve.gov/econresdata/scf/files/'+year+'_scf'+year[2:]+'bs.zip'
    
    else: 
        url = 'http://www.federalreserve.gov/econresdata/scf/files/scfp'+year+'s.zip'    
        
    url = requests.get(url)
    url_unzipped = zipfile.ZipFile(io.BytesIO(url.content))
    return url_unzipped.extract(url_unzipped.namelist()[0])

df1983 = pd.read_stata(unzip_survey_file(year = '1983'))
df1992 = pd.read_stata(unzip_survey_file(year = '1992'))
df2001 = pd.read_stata(unzip_survey_file(year = '2001'))

"""
There is no Summary Extract dataset for 1983, so we'll rename the variable names in the 1983 Full 
dataset so that they correspond to the variable names in the other survey years.

Also, 161 out of the 4262 total households covered in the 1983 survey actually reported having 
negative income. This isn't the case for the other survey years we are considering, and it 
complicates our analysis a bit below. Because of this, we drop any obs. that report negative 
incomes before proceeding. This has a near-zero impact on any of our results, since all but 2 
of these observations recieve a weight of zero. The two non-zero weight observations reporting
negative incomes account for only <0.05% of the total population, so not much is lost be 
excluding them.

Going forward: it might be worthwhile to figure out why there are instances of negative incomes
in the 1983 survey yet none for the other years. 
"""
df1983 = df1983.rename(columns = {'b3201':'income', 'b3324':'networth', 'b3015' : 'wgt'})

df1983 = df1983[df1983['income']>=0]



def weighted_percentiles(data, variable, weights, percentiles = [], 
                         dollar_amt = False, subgroup = None, limits = []):
    """
    data               specifies what dataframe we're working with
    
    variable           specifies the variable name (e.g. income, networth, etc.) in the dataframe
    
    percentiles = []   indicates what percentile(s) to return (e.g. 90th percentile = .90)
    
    weights            corresponds to the weighting variable in the dataframe
    
    dollar_amt = False returns the percentage of total income earned by that percentile 
                       group (i.e. bottom 80% of earners earned XX% of total)
                         
    dollar_amt = True  returns the $ amount earned by that percentile (i.e. 90th percentile
                       earned $X)
                         
    subgroup = ''      isolates the analysis to a particular subgroup in the dataset. For example
                       subgroup = 'age' would return the income distribution of the age group 
                       determined by the limits argument
                       
    limits = []        Corresponds to the subgroup argument. For example, if you were interesting in 
                       looking at the distribution of income across heads of household aged 18-24,
                       then you would input "subgroup = 'age', limits = [18,24]"
                         
    """
    import numpy 
    a  = list()
    data[variable+weights] = data[variable]*data[weights]
    if subgroup is None:
        tt = data
    else:
        tt = data[data[subgroup].astype(int).isin(range(limits[0],limits[1]+1))] 
    values, sample_weight = tt[variable], tt[weights]
    
    for index in percentiles: 
        values = numpy.array(values)
        index = numpy.array(index)
        sample_weight = numpy.array(sample_weight)

        sorter = numpy.argsort(values)
        values = values[sorter]
        sample_weight = sample_weight[sorter]

        weighted_percentiles = numpy.cumsum(sample_weight) - 0.5 * sample_weight
        weighted_percentiles /= numpy.sum(sample_weight)
        a.append(numpy.interp(index, weighted_percentiles, values))
    
    if dollar_amt is False:    
        return[tt.loc[tt[variable]<=a[x],
                      variable+weights].sum()/tt[variable+weights].sum() for x in range(len(percentiles))]
    else:
        return a


get_ipython().magic('pylab inline')
import matplotlib.pyplot as plt                            

def figureprefs(data, variable = 'income', labels = False, legendlabels = []):
    
    percentiles = [i * 0.05 for i in range(20)]+[0.99, 1.00]

    fig, ax = plt.subplots(figsize=(10,8));

    ax.set_xticks([i*0.1 for i in range(11)]);       #Sets the tick marks
    ax.set_yticks([i*0.1 for i in range(11)]);

    vals = ax.get_yticks()                           #Labels the tick marks
    ax.set_yticklabels(['{:3.0f}%'.format(x*100) for x in vals]);
    ax.set_xticklabels(['{:3.0f}%'.format(x*100) for x in vals]);

    ax.set_title('Lorenz Curve: United States, 1983 vs. 2013',  #Axes titles
                  fontsize=18, loc='center');
    ax.set_ylabel('Cumulative Percent of Total Income', fontsize = 12);
    ax.set_xlabel('Percent of Familes Ordered by Incomes', fontsize = 12);
    
    if type(data) == list:
        values = [weighted_percentiles(data[x], variable,
                    'wgt', dollar_amt = False, percentiles = percentiles) for x in range(len(data))]
        for index in range(len(data)):
            plt.plot(percentiles,values[index],
                     linewidth=2.0, marker = 's',clip_on=False,label=legendlabels[index]);
            for num in [10, 19, 20]:
                ax.annotate('{:3.1f}%'.format(values[index][num]*100), 
                    xy=(percentiles[num], values[index][num]),
                    ha = 'right', va = 'center', fontsize = 12);

    else:
        values = weighted_percentiles(data, variable,
                    'wgt', dollar_amt = False, percentiles = percentiles)
        plt.plot(percentiles,values,
                     linewidth=2.0, marker = 's',clip_on=False,label=legendlabels);

    plt.plot(percentiles,percentiles, linestyle =  '--', color='k',
            label='Perfect Equality');
   
    legend(loc = 2)

    

years_graph = [df2013, df1983]
labels = ['2013', '1983']

figureprefs(years_graph, variable = 'income', legendlabels = labels);


"""
Note: All Summary Extract data for survey years 1989 and later have been adjusted for inflation
(2013=100). This isn't the case for survey data prior to 1989, so we'll have to adjust the 1983 
data manually.
"""

from pandas.io import wb                                            # World Bank api

import warnings
warnings.simplefilter(action = "ignore", category = FutureWarning)  #Ignore these two lines


cpi = wb.download(indicator='FP.CPI.TOTL' , country= 'USA', start=1983, end=2013)  #CPI

"""
The World Bank CPI series is indexed so that 2010 = 100. We'll have to re-index it so that 2013 = 100
to be consistent with the other data.
"""
cpi1983 = (100/cpi['FP.CPI.TOTL'][2013-2013])*cpi['FP.CPI.TOTL'][2013-1983]/100
df1983['realincome'] = df1983['income']/cpi1983



percentiles = [i * 0.01 for i in range(1,100)]+[0.99]+[0.999]

incomes = pd.DataFrame({'2001': weighted_percentiles(df2001, 'income', 'wgt', dollar_amt = True, percentiles =percentiles),
'2013': weighted_percentiles(df2013, 'income', 'wgt', dollar_amt = True, percentiles = percentiles),
'1992': weighted_percentiles(df1992, 'income', 'wgt', dollar_amt = True, percentiles = percentiles),
'1983': weighted_percentiles(df1983, 'realincome', "wgt", dollar_amt = True, percentiles = percentiles)})

fig, ax = plt.subplots(figsize=(10,6))
plt.plot(percentiles,(incomes['2013']-incomes['1983'])/incomes['1983']/(2013-1983+1),
         linewidth = 2.0, label = '1983-2013');
yvals = ax.get_yticks()
ax.set_xticks([i * 0.1 for i in range(11)])
xvals = ax.get_xticks()
ax.set_yticklabels(['{:3.2f}%'.format(x*100) for x in yvals]);
ax.set_xticklabels(['{:3.0f}'.format(x*100) for x in xvals]);
ax.set_title('Annual real income growth by income percentile',  #Axes titles
                  fontsize=18, loc='center');
ax.axhline(y=0,xmin = 0, xmax = 1, linestyle = '--', color = 'k');
ax.set_ylabel('Average annual growth rate of real income');
ax.set_xlabel('Income percentile');
legend(loc=2);

fig, ax = plt.subplots(figsize=(10,6))
plt.plot(percentiles,(incomes['2001']-incomes['1992'])/incomes['1992']/(2001-1992+1),
         linewidth = 2.0, label = '1992-2001');
plt.plot(percentiles,(incomes['2013']-incomes['1983'])/incomes['1983']/(2013-1983+1),
         linewidth = 2.0, label = '1983-2013');
yvals = ax.get_yticks()
ax.set_xticks([i * 0.1 for i in range(11)])
xvals = ax.get_xticks()
ax.set_yticklabels(['{:3.2f}%'.format(x*100) for x in yvals]);
ax.set_xticklabels(['{:3.0f}'.format(x*100) for x in xvals]);
ax.set_title('Annual real income growth by income percentile',  #Axes titles
                  fontsize=18, loc='center');
ax.axhline(y=0,xmin = 0, xmax = 1, linestyle = '--', color = 'k');
ax.set_ylabel('Average annual growth rate of real income');
ax.set_xlabel('Income percentile');
legend(loc=2);

