get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
data = pd.read_csv("Electric_Production.csv",index_col=0)
data.head()

data.index

data.index = pd.to_datetime(data.index)

data.head()

data.index

data[pd.isnull(data['IPG2211A2N'])]

data.columns = ['Energy Production']

data.head()

import plotly
# plotly.tools.set_credentials_file()

from plotly.plotly import plot_mpl
from statsmodels.tsa.seasonal import seasonal_decompose
result = seasonal_decompose(data, model='multiplicative')
fig = result.plot()
plot_mpl(fig)

import plotly.plotly as ply
import cufflinks as cf
# Check the docs on setting up offline plotting

data.iplot(title="Energy Production Jan 1985--Jan 2018", theme='pearl')



from pyramid.arima import auto_arima

stepwise_model = auto_arima(data, start_p=1, start_q=1,
                           max_p=3, max_q=3, m=12,
                           start_P=0, seasonal=True,
                           d=1, D=1, trace=True,
                           error_action='ignore',  
                           suppress_warnings=True, 
                           stepwise=True) 

stepwise_model.aic()

data.head()

data.info()

train = data.loc['1985-01-01':'2016-12-01']

train.tail()

test = data.loc['2015-01-01':]

test.head()

test.tail()

len(test)

stepwise_model.fit(train)

future_forecast = stepwise_model.predict(n_periods=37)

future_forecast

future_forecast = pd.DataFrame(future_forecast,index = test.index,columns=['Prediction'])

future_forecast.head()

test.head()

pd.concat([test,future_forecast],axis=1).iplot()

future_forecast2 = future_forcast

pd.concat([data,future_forecast2],axis=1).iplot()



