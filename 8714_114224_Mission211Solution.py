import pandas

board_games = pandas.read_csv("board_games.csv")
board_games = board_games.dropna(axis=0)
board_games = board_games[board_games["users_rated"] > 0]

board_games.head()

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt

plt.hist(board_games["average_rating"])

print(board_games["average_rating"].std())
print(board_games["average_rating"].mean())

from sklearn.cluster import KMeans

clus = KMeans(n_clusters=5)
cols = list(board_games.columns)
cols.remove("name")
cols.remove("id")
cols.remove("type")
numeric = board_games[cols]

clus.fit(numeric)

import numpy
game_mean = numeric.apply(numpy.mean, axis=1)
game_std = numeric.apply(numpy.std, axis=1)

labels = clus.labels_

plt.scatter(x=game_mean, y=game_std, c=labels)

correlations = numeric.corr()

correlations["average_rating"]

from sklearn.linear_model import LinearRegression

reg = LinearRegression()
cols.remove("average_rating")
cols.remove("bayes_average_rating")
reg.fit(board_games[cols], board_games["average_rating"])
predictions = reg.predict(board_games[cols])

numpy.mean((predictions - board_games["average_rating"]) ** 2)



