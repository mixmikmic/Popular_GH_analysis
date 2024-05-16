from bat.log_to_dataframe import LogToDataFrame
from bat.utils import plot_utils

# Just some plotting defaults
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
plot_utils.plot_defaults()

# Convert it to a Pandas DataFrame
http_df = LogToDataFrame('../data/http.log')
http_df.head()

http_df[['request_body_len','response_body_len']].hist()

http_df['uid'].resample('1S').count().plot()
plt.xlabel('HTTP Requests per Second')

