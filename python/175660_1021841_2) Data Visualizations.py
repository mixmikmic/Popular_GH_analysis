# import Useful libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

colors = sns.color_palette()

# load datasets and store into variables
anime = pd.read_csv('anime.csv')
rating = pd.read_csv('rating.csv')

anime = anime.dropna()
plt.figure(figsize=(10,8))

for i, col in enumerate(anime.type.unique()):
    ax = plt.subplot(3, 2, i + 1)
    plt.yticks([.5, .4, .3, .2, .1, 0])
    plt.ylim(ymax=.6)
    sns.kdeplot(anime[anime['type']==col].rating, shade=True, label=col, color = colors[i])   

rating.rating.replace({-1: np.nan}, regex=True, inplace = True)

anime_tv = anime[anime['type']=='TV']
anime_tv.head()

# join the two dataframes on the anime_id columns
merged = rating.merge(anime_tv, left_on = 'anime_id', right_on = 'anime_id', suffixes= ['_user', ''])
merged.rename(columns = {'rating_user':'user_rating'}, inplace = True)

# reogranize the dataframe to the desired columns
merged = merged[['user_id', 'anime_id', 'name', 'genre', 'type', 'episodes', 'user_rating', 'rating']]
merged.head()

# Lets see how the critc review varies from the users' for a given show

merged[merged['name']=='One Punch Man'].user_rating.plot(kind='hist', color = colors[0], alpha=.7)
plt.axvline(merged[merged['name']=='One Punch Man'].rating.mean(), color=colors[0], linestyle='dashed', linewidth=2, label='Critic Rating')

plt.xlabel('Audience Rating')
plt.title('One Punch Man User Rating Histogram')
plt.legend(loc = 'upper left')

# Shows with the highest count of 10 star ratings
highest_10_count = merged[merged.user_rating == 10].groupby('name').rating.count().sort_values(ascending = False)

highest_10_count[:10 -1].plot(kind='bar', width = 0.8,color = colors[0], alpha=0.7)
plt.title('Count of 10* Ratings')

# Series of average rating per user
user_rating_mean = merged.groupby('user_id').user_rating.mean().dropna()

sns.kdeplot(user_rating_mean, shade=True, color = colors[3], label='User Rating') 
plt.xticks([2,4,6,8,10])
plt.xlim(0,11)
plt.title("Density of Users' Average Rating")
plt.xlabel('Rating')

# series of user standard deviations
user_std = merged.groupby('user_id').user_rating.std().dropna()

sns.kdeplot(user_std, shade=True, color = colors[2], label='User Std') 
plt.xlim(-1,5)
plt.title('Density of User Standard Deviations')
plt.xlabel('Standard Deviation')

# Series of user rating counts
user_rating_count = rating.dropna().groupby('user_id').rating.count()

user_rating_count.plot(kind='kde', color=colors[1])
plt.axvline(rating.dropna().groupby('user_id').rating.count().mean(), color=colors[1], linestyle='dashed', linewidth=2)
plt.xlim(-100, 400)
plt.xlabel('Rating Count')
plt.title('Density of Ratings Count Per User')



