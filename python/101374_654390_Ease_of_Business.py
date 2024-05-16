get_ipython().magic('matplotlib inline')
import numpy as np
import pandas as pd
np.random.seed(45)
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

indicator = pd.read_csv('indicator1.csv')
indicator.head()

indicator.drop(indicator.columns[[0, 2]], axis=1, inplace=True)
indicator.head()

indicator.info()

indicator.describe()

indicator.hist(bins=50, figsize=(20, 15))
plt.savefig('numeric_attributes.png')
plt.show()

from pandas.plotting import scatter_matrix

attributes = ["GDP_per_capita", "Hours_do_tax", "Days_reg_bus", "Cost_start_Bus",
              "Bus_tax_rate", "Ease_Bus"]
scatter_matrix(indicator[attributes], figsize=(12, 8))
plt.savefig("scatter_matrix_plot.png")
plt.show()

from sklearn.linear_model import LinearRegression
X = indicator.drop(['country', 'Ease_Bus'], axis=1)
regressor = LinearRegression()
regressor.fit(X, indicator.Ease_Bus)

print('Estimated intercept coefficient:', regressor.intercept_)

print('Number of coefficients:', len(regressor.coef_))

pd.DataFrame(list(zip(X.columns, regressor.coef_)), columns = ['features', 'est_coef'])

indicator.plot(kind="scatter", x="Days_reg_bus", y="Ease_Bus",
             alpha=0.8)
plt.savefig('scatter_plot.png')

from sklearn.cross_validation import train_test_split
y = indicator.Ease_Bus

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

regressor.score(X_test, y_test)

from sklearn.metrics import mean_squared_error
regressor_mse = mean_squared_error(y_pred, y_test)

import math
math.sqrt(regressor_mse)

regressor.predict([[41096.157300, 5.0, 3, 58.7, 161.0]])

indicator.loc[indicator['country'] == 'Belgium']

regressor.predict([[42157.927990, 0.4, 2, 21.0, 131.0]])

indicator.loc[indicator['country'] == 'Canada']

plt.scatter(regressor.predict(X_train), regressor.predict(X_train)-y_train, c='indianred', s=40)
plt.scatter(regressor.predict(X_test), regressor.predict(X_test)-y_test, c='b', s=40)
plt.hlines(y=0, xmin=0, xmax=200)
plt.title('Residual plot using training(red) and test(blue) data')
plt.ylabel('Residual')
plt.savefig('residual_plot.png')

