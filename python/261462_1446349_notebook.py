# Import modules
import pandas as pd

# Read names into a dataframe: bnames
bnames = pd.read_csv('datasets/names.csv.gz')
bnames.head()

# bnames_top5: A dataframe with top 5 popular male and female names for the decade
import numpy as np
bnames_2010 = bnames.loc[bnames['year'] > 2010]
bnames_2010_agg = bnames_2010.groupby(['sex', 'name'], as_index=False)['births'].sum()
bnames_top5 = bnames_2010_agg.sort_values(['sex', 'births'], ascending=[True, False]).groupby('sex').head().reset_index(drop=True)
print(bnames_top5.head())

# Import modules
import pandas as pd

# Read names into a dataframe: bnames
bnames = pd.read_csv('datasets/names.csv.gz')
bnames2 = bnames.copy()
# Compute the proportion of births by year and add it as a new column
total_births_by_year = bnames.groupby('year')['births', 'year'].transform(sum)
bnames2['prop_births'] = bnames2.births/ total_births_by_year.births
print(bnames2)

# Set up matplotlib for plotting in the notebook.
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
# Import modules
import pandas as pd

# Read names into a dataframe: bnames
bnames = pd.read_csv('datasets/names.csv.gz')
def plot_trends(name, sex):
  data = bnames[(bnames.name == name) & (bnames.sex == sex)]
  ax = data.plot(x = "year", y = "births")
  ax.set_xlim(1880, 2016)
  return ax


# Plot trends for Elizabeth and Deneen 
for name in ('Elizabeth', 'Deneen'):
    plt.axis = plot_trends(name, 'F')
plt.xlabel('Year')
plt.ylabel('Births')
plt.show()
# How many times did these female names peak?
num_peaks_elizabeth = 3
num_peaks_deneen    = 1

# top10_trendy_names | A Data Frame of the top 10 most trendy names
names = pd.DataFrame()
name_and_sex_grouped = bnames.groupby(['name', 'sex'])
names['total'] = name_and_sex_grouped['births'].sum()
names['max'] = name_and_sex_grouped['births'].max()
names['trendiness'] = names['max']/names['total']

top10_trendy_names = names.loc[names['total'] > 999].sort_values(['trendiness'], ascending=False).head(10).reset_index()

print(top10_trendy_names)

# Read lifetables from datasets/lifetables.csv
lifetables = pd.read_csv('datasets/lifetables.csv')

# Extract subset relevant to those alive in 2016
lifetables_2016 = lifetables[lifetables['year'] + lifetables['age'] == 2016]

# Plot the mortality distribution: year vs. lx
lifetables_2016.plot(x= 'year', y= 'lx')
plt.show()
lifetables_2016.head()

# Create smoothened lifetable_2016_s by interpolating values of lx
year = np.arange(1900, 2016)
mf = {"M": pd.DataFrame(), "F": pd.DataFrame()}
for sex in ["M", "F"]:
  d = lifetables_2016[lifetables_2016['sex'] == sex][["year", "lx"]]
  mf[sex] = d.set_index('year').reindex(year).interpolate().reset_index()
  mf[sex]['sex'] = sex

lifetable_2016_s = pd.concat(mf, ignore_index = True)
lifetable_2016_s.head()
print(lifetable_2016_s)

def get_data(name, sex):
    name_sex = ((bnames['name'] == name) & 
                (bnames['sex'] == sex))
    data = bnames[name_sex].merge(lifetable_2016_s)
    data['n_alive'] = data['lx']/(10**5)*data['births']
    return data
    

def plot_data(name, sex):
    fig, ax = plt.subplots()
    dat = get_data(name, sex)
    dat.plot(x = 'year', y = 'births', ax = ax, 
               color = 'black')
    dat.plot(x = 'year', y = 'n_alive', 
              kind = 'area', ax = ax, 
              color = 'steelblue', alpha = 0.8)
    ax.set_xlim(1900, 2016)
    return
# Plot the distribution of births and number alive for Joseph and Brittany
    
    plot_data('Britanny', 'F')
    plot_data('Joseph', 'M')

# Import modules
from wquantiles import quantile

def estimate_age(name, sex):
    data = get_data(name, sex)
    qs = [0.75, 0.5, 0.25]
    quantiles = [2016 - int(quantile(data.year, data.n_alive, q)) for q in qs]
    result = dict(zip(['q25', 'q50', 'q75'], quantiles))
    result['p_alive'] = round(data.n_alive.sum()/data.births.sum()*100, 2)
    result['sex'] = sex
    result['name'] = name
    return pd.Series(result)
# Estimate the age of Gertrude
estimate_age('Gertrude', 'F')

# Import modules
from wquantiles import quantile
import pandas as pd
import numpy as np
bnames = pd.read_csv('datasets/names.csv.gz')
# Function to estimate age quantiles
# Create smoothened lifetable_2016_s by interpolating values of lx
# Read lifetables from datasets/lifetables.csv
lifetables = pd.read_csv('datasets/lifetables.csv')

# Extract subset relevant to those alive in 2016
lifetables_2016 = lifetables[lifetables['year'] + lifetables['age'] == 2016]
year = np.arange(1900, 2016)
mf = {"M": pd.DataFrame(), "F": pd.DataFrame()}
for sex in ["M", "F"]:
  d = lifetables_2016[lifetables_2016['sex'] == sex][["year", "lx"]]
  mf[sex] = d.set_index('year').reindex(year).interpolate().reset_index()
  mf[sex]['sex'] = sex
lifetable_2016_s = pd.concat(mf, ignore_index = True)

# Function to estimate age quantiles

def get_data(name, sex):
    name_sex = ((bnames['name'] == name) & 
                (bnames['sex'] == sex))
    data = bnames[name_sex].merge(lifetable_2016_s)
    data['n_alive'] = data['lx']/(10**5)*data['births']
    return data
def estimate_age(name, sex):
    data = get_data(name, sex)
    qs = [0.75, 0.5, 0.25]
    quantiles = [2016 - int(quantile(data.year, data.n_alive, q)) for q in qs]
    result = dict(zip(['q25', 'q50', 'q75'], quantiles))
    result['p_alive'] = round(data.n_alive.sum()/data.births.sum()*100, 2)
    result['sex'] = sex
    result['name'] = name
    return pd.Series(result)
# Estimate the age of Gertrude
estimate_age('Gertrude', 'F')
top_10_female_names = bnames.groupby(['name', 'sex'], as_index = False).agg({'births': np.sum}).sort_values('births', ascending = False).query('sex == "F"').head(10).reset_index(drop = True)
estimates = pd.concat([estimate_age(name, 'F') for name in top_10_female_names.name], axis = 1)
median_ages = estimates.T.sort_values('q50').reset_index(drop = True)

