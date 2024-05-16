import pandas as pd
police_killings = pd.read_csv("police_killings.csv", encoding="ISO-8859-1")
police_killings.head(5)

police_killings.columns

counts = police_killings["raceethnicity"].value_counts()

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt

plt.bar(range(6), counts)
plt.xticks(range(6), counts.index, rotation="vertical")

counts / sum(counts)

police_killings["p_income"][police_killings["p_income"] != "-"].astype(float).hist(bins=20)

police_killings["p_income"][police_killings["p_income"] != "-"].astype(float).median()

state_pop = pd.read_csv("state_population.csv")

counts = police_killings["state_fp"].value_counts()

states = pd.DataFrame({"STATE": counts.index, "shootings": counts})

states = states.merge(state_pop, on="STATE")

states["pop_millions"] = states["POPESTIMATE2015"] / 1000000
states["rate"] = states["shootings"] / states["pop_millions"]

states.sort("rate")

police_killings["state"].value_counts()

pk = police_killings[
    (police_killings["share_white"] != "-") & 
    (police_killings["share_black"] != "-") & 
    (police_killings["share_hispanic"] != "-")
]

pk["share_white"] = pk["share_white"].astype(float)
pk["share_black"] = pk["share_black"].astype(float)
pk["share_hispanic"] = pk["share_hispanic"].astype(float)

lowest_states = ["CT", "PA", "IA", "NY", "MA", "NH", "ME", "IL", "OH", "WI"]
highest_states = ["OK", "AZ", "NE", "HI", "AK", "ID", "NM", "LA", "CO", "DE"]

ls = pk[pk["state"].isin(lowest_states)]
hs = pk[pk["state"].isin(highest_states)]

columns = ["pop", "county_income", "share_white", "share_black", "share_hispanic"]

ls[columns].mean()

hs[columns].mean()

