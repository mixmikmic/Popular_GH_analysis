import pandas as pd
import numpy as np
from fbprophet import Prophet

get_ipython().magic('matplotlib inline')

# Read in the source data - downloaded from google analytics
df = pd.read_excel('https://github.com/chris1610/pbpython/blob/master/data/All-Web-Site-Data-Audience-Overview.xlsx?raw=True')

df.head()

# Convert to log format
df['Sessions'] = np.log(df['Sessions'])

# Need to name the columns like this in order for prophet to work
df.columns = ["ds", "y"]

# Create the model
m1 = Prophet()
m1.fit(df)

# Predict out a year
future1 = m1.make_future_dataframe(periods=365)
forecast1 = m1.predict(future1)

m1.plot(forecast1);

m1.plot_components(forecast1);

articles = pd.DataFrame({
  'holiday': 'publish',
  'ds': pd.to_datetime(['2014-09-27', '2014-10-05', '2014-10-14', '2014-10-26', '2014-11-9',
                        '2014-11-18', '2014-11-30', '2014-12-17', '2014-12-29', '2015-01-06',
                        '2015-01-20', '2015-02-02', '2015-02-16', '2015-03-23', '2015-04-08',
                        '2015-05-04', '2015-05-17', '2015-06-09', '2015-07-02', '2015-07-13',
                        '2015-08-17', '2015-09-14', '2015-10-26', '2015-12-07', '2015-12-30',
                        '2016-01-26', '2016-04-06', '2016-05-16', '2016-06-15', '2016-08-23',
                        '2016-08-29', '2016-09-06', '2016-11-21', '2016-12-19', '2017-01-17',
                        '2017-02-06', '2017-02-21', '2017-03-06']),
  'lower_window': 0,
  'upper_window': 5,
})
articles.head()

m2 = Prophet(holidays=articles).fit(df)
future2 = m2.make_future_dataframe(periods=365)
forecast2 = m2.predict(future2)
m2.plot(forecast2);

m2.plot_components(forecast2);

forecast2.head()

forecast2["Sessions"] = np.exp(forecast2.yhat).round()

forecast2[["ds", "Sessions"]].tail()



