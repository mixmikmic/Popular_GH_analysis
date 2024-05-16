get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

import entrofy.mappers
import entrofy.core

seed = 20160415

df=pd.read_csv('test.csv')
df.head()

age = np.random.poisson(30.0, size=len(df.index))
df["age"] = age
df.head()

ax = entrofy.plotting.plot_distribution(df, "subfield",
                                        xtype="categorical",
                                        cmap="YlGnBu", ax=None)

ax = entrofy.plotting.plot_distribution(df, "age",
                                        xtype="continuous",
                                        cmap="YlGnBu", bins=20)

fig, (ax1, ax2) = plt.subplots(1,2, figsize=(16,6))
ax1 = entrofy.plotting.plot_distribution(df, "gender",
                                         xtype="categorical", ax=ax1)
ax2 = entrofy.plotting.plot_distribution(df, "age",
                                         xtype="continuous", ax=ax2)

ax = entrofy.plotting.plot_correlation(df, "subfield", "career_stage",
                                       xtype="categorical", 
                                       ytype="categorical", s=5, prefac=5,
                                       cmap="YlGnBu")

plottypes = ["box", "strip", "violin"]#, "swarm"]

fig, axes = plt.subplots(1, len(plottypes), figsize=(20,5))
for ax, t in zip(axes, plottypes):
    ax = entrofy.plotting.plot_correlation(df, "subfield", "age",
                                           xtype="categorical", 
                                           ytype="continuous",
                                           cat_type=t, ax=ax)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,6))

ax1 = entrofy.plotting.plot_correlation(df, "age", "age", ax=ax1,
                                        xtype="continuous",
                                        ytype="continuous", cont_type="kde")
ax2 = entrofy.plotting.plot_correlation(df, "age", "age", ax=ax2,
                                        xtype="continuous",
                                        ytype="continuous", cont_type="scatter")

weights = {"gender": 1.0, "career_stage": 1.0, 
           "country": 1.0, "subfield": 1.0}

fig, axes = entrofy.plotting.plot_triangle(df, weights)

weights = {"gender": 1.0, "career_stage": 1.0, 
           "country": 1.0, "subfield": 1.0, "age": 1.0}

mappers = entrofy.core.construct_mappers(df, weights)

mappers

datatypes = {"age": "continuous",
             "gender": "categorical",
             "subfield": "categorical",
             "country": "categorical",
             "career_stage": "categorical"}

mappers = entrofy.core.construct_mappers(df, weights, datatypes)

mappers

mappers["age"].boundaries

mappers["subfield"].targets

# new targets in alphabetical order
new_targets = [0.2, 0.5, 0.2, 0.1]

# sort keys for the targets dictionary alphabetically:
sorted_keys = np.sort(mappers["subfield"].targets.keys())

for t, key in zip(new_targets, sorted_keys):
    mappers["subfield"].targets[key] = t

mappers["subfield"].targets

gender_targets = {"female": 0.4, "male": 0.4, "other": 0.2}
gender_mapper = entrofy.mappers.ObjectMapper(df["gender"], n_out=3, targets=gender_targets)

gender_mapper.targets

career_mapper = entrofy.mappers.ObjectMapper(df["career_stage"])

career_mapper.targets

career_mapper.targets["grad-student"] = 0.5
career_mapper.targets["junior-faculty"] = 0.3
career_mapper.targets["senior-faculty"] = 0.2

career_mapper.targets

country_targets = {"OECD": 0.7, "non-OECD": 0.3}
country_mapper = entrofy.mappers.ObjectMapper(df["country"], n_out=2, targets=country_targets)

subfield_mapper = entrofy.mappers.ObjectMapper(df["subfield"])

subfield_mapper.targets

subfield_mapper.targets["cosmology"] = 0.5
subfield_mapper.targets["astrophysics"] = 0.2
subfield_mapper.targets["exoplanets"] = 0.2
subfield_mapper.targets["solar"] = 0.1

age_mapper = entrofy.mappers.ContinuousMapper(df["age"], n_out=3)

age_mapper.boundaries

age_mapper.targets

age_boundaries = [0.0, 20.0, 30.0, 40.0, 50.0]
age_targets = {"0-20":0.3, "20-30":0.3, "30-40":0.2, "40-50":0.2}
age_column_names = ["0-20", "20-30", "30-40", "40-50"]
age_mapper = entrofy.mappers.ContinuousMapper(df["age"], n_out=4,
                                             boundaries=age_boundaries,
                                             targets=age_targets, column_names = age_column_names)

print(age_mapper.boundaries)
print(age_mapper.targets)

mappers = {"age": age_mapper, "gender": gender_mapper, 
          "country": country_mapper, "subfield": subfield_mapper, 
          "career_stage": career_mapper}

idx, max_score = entrofy.core.entrofy(df, 20,
                                      mappers=mappers,
                                      weights=weights,
                                      seed=seed)
print(max_score)

df.loc[idx]

mappers

mappers["age"].targets

# plot the input data distribution
fig, ax  = entrofy.plotting.plot_triangle(df, weights,
                                          mappers=mappers,
                                          cat_type="violin")

df_out = df.loc[idx]

# plot distribution of selected:
fig, ax  = entrofy.plotting.plot_triangle(df_out, weights,
                                          mappers=mappers,
                                          cat_type="violin")

_, _ = entrofy.plotting.plot_fractions(df["subfield"], idx,
                                       "subfield", mappers["subfield"])

_ = entrofy.plotting.plot(df, idx, weights, mappers=mappers, cols=4)

idx

opt_outs = [idx[3], idx[10], idx[16]]
pre_selects = idx.drop(opt_outs)

n = len(opt_outs) + len(pre_selects)
print("The total number of participants is " + str(n))

new_idx, max_score = entrofy.core.entrofy(df, n=n, mappers=mappers, weights=weights,
                     pre_selects=list(pre_selects), opt_outs=opt_outs, seed=seed)

df.loc[new_idx]

pre_selects = []
opt_outs = []

entrofy.core.save(idx, "test_run1.pkl", dataframe=df, mappers=mappers, 
                 weights=weights, pre_selects=pre_selects, opt_outs=opt_outs)

state = entrofy.core.load("test_run1.pkl", dataframe=df)

state.keys()

