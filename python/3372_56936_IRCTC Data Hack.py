get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Load the data into a dataframe
df = pd.read_csv("data/isl_wise_train_detail_03082015_v1.csv")

sns.set_context("poster")
# Show some rows
df.head()

df.columns

# Convert time columns to datetime objects
df[u'Arrival time'] = pd.to_datetime(df[u'Arrival time'])
df[u'Departure time'] = pd.to_datetime(df[u'Departure time'])

df.head()

fig, ax = plt.subplots(1,2, sharey=True)
df[u'Arrival time'].map(lambda x: x.hour).hist(ax=ax[0], bins=24)
df[u'Departure time'].map(lambda x: x.hour).hist(ax=ax[1], bins=24)
ax[0].set_xlabel("Arrival Time")
ax[1].set_xlabel("Departure Time")

df["Stoppage"] = (df[u'Departure time'] - df[u'Arrival time']).astype('timedelta64[m]') # Find stoppage time in minutes
# Plot distribution of stoppage time
df["Stoppage"].hist()
plt.xlabel("Stoppage Time")

df["Stoppage"][(df["Stoppage"]> 0) & (df["Stoppage"] < 61)].hist() # Let us take that max stoppage time can be an hour. 
plt.xlabel("Stoppage Time")

df["Stoppage"][(df["Stoppage"]> 0) & (df["Stoppage"] < 31)].hist(bins=30) # Let us take that max stoppage time can be an hour. 
plt.xlabel("Stoppage Time")

df_stoppage_30 = df[(df["Stoppage"]> 0) & (df["Stoppage"] < 31)] # Filter data between nice stoppage times
# Plot data for this stoppage time range.
fig, ax = plt.subplots(1,2, sharey=True)
df_stoppage_30[u'Arrival time'].map(lambda x: x.hour).hist(ax=ax[0], bins=24)
df_stoppage_30[u'Departure time'].map(lambda x: x.hour).hist(ax=ax[1], bins=24)
ax[0].set_xlabel("Arrival Time")
ax[1].set_xlabel("Departure Time")

# Total Number of stations of the train, last arrival time, first departure time, last distance, first station and last station.

df_train_dist = df[[u'Train No.', u'station Code', u'Arrival time', u'Departure time',
                    u'Distance', u'Source Station Code', u'Destination station Code']]\
.groupby(u'Train No.').agg({u'station Code': "count", u'Arrival time': "last",
                                                               u'Departure time': "first", u'Distance': "last",
                                                               u'Source Station Code': "first", u'Destination station Code': "last"})

df_train_dist.head()

# Let us plot the distribution of the distances as well as station codes, as well as arrival and departure times
fig, ax = plt.subplots(2,2)
df_train_dist[u'station Code'].hist(ax=ax[0][0], bins=range(df_train_dist[u'station Code'].max() + 1))
df_train_dist[u'Distance'].hist(ax=ax[0][1], bins=50)
ax[1][0].set_xlabel("Total Stations stopped")
ax[1][1].set_xlabel("Total Distance covered")

df_train_dist[u'Arrival time'].map(lambda x: x.hour).hist(ax=ax[1][0], bins=range(24))
df_train_dist[u'Departure time'].map(lambda x: x.hour).hist(ax=ax[1][1], bins=range(24))
ax[1][0].set_xlabel("Arrival Time")
ax[1][1].set_xlabel("Departure Time")

sns.lmplot(x=u'station Code', y=u'Distance', data=df_train_dist, x_estimator=np.mean)

# Lets us see what are some general statistics of the distances and the number of stops. 
df_train_dist.describe()

df[[u'Train No.', u'Station Name']].groupby(u'Station Name').count().sort(u'Train No.', ascending=False).head(20)

df[[u'Train No.', u'Station Name']].groupby(u'Station Name').count().hist(bins=range(1,320,2), log=True)
plt.xlabel("Number of trains stopping")
plt.ylabel("Number of stations")

df_train_dist.sort(u'station Code', ascending=False).head(10) # Top 10 trains with maximum number of stops

df_train_dist.sort(u'Distance', ascending=False).head(10) # Top 10 trains with maximum distance

fig, ax = plt.subplots(1,2)
sns.regplot(x=df_train_dist[u'Arrival time'].map(lambda x: x.hour), y=df_train_dist[u'Distance'], x_estimator=np.mean, ax=ax[0])
sns.regplot(x=df_train_dist[u'Departure time'].map(lambda x: x.hour), y=df_train_dist[u'Distance'], x_estimator=np.mean, ax=ax[1])



