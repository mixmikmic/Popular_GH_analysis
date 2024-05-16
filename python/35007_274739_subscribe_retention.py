import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from sklearn.linear_model import LinearRegression
get_ipython().magic('matplotlib inline')

subscriptions = pd.read_csv("subscription.csv",index_col='user_id')

# 'subscription_signup_date' is always Jan, 2015 in this table. useless, so delete it
del subscriptions['subscription_signup_date']

# rename some long column name to short ones, which is easier to read
subscriptions.rename(columns={'subscription_monthly_cost':'monthly_cost',
                              'billing_cycles':'bill_cycles'},inplace=True)

# check the data, have a feeling about it
subscriptions.sample(10)

count_by_cost = subscriptions.groupby('monthly_cost').apply(lambda df: df.bill_cycles.value_counts()).unstack()
# for index 'n', the value is the #people who paid 'n' billing cycles
count_by_cost

# for each row in 'count_by_cost', we perform a reverse cumsum to get the #people by the end of each billing cycles
total_by_cost = count_by_cost.apply(lambda s: s.iloc[::-1].cumsum().iloc[::-1],axis=1).transpose()
total_by_cost

total_by_cost.plot()

def make_time_features(t):
    """
    three features:
    1. t: #cycles
    2. t-square: square of #cycles
    3. logt: log(#cycles)
    """
    return pd.DataFrame({'t': t,'logt': np.log(t),'tsquare':t*t },index = t)

def fit_linear_regression(s):
    """
    target:
    log(s): s is #subscribers left by the end of each billing cycle
    do this transformation, to guarantee that, after tranforming back, the fitted result is always positive
    """
    X = make_time_features(s.index)
    return LinearRegression().fit(X,np.log(s))

lr_by_cost = total_by_cost.apply(fit_linear_regression,axis=0)

allt = np.arange(1,13)
Xwhole = make_time_features(allt)
Xwhole

# call each cost's model to fit on above features
predicts = lr_by_cost.apply(lambda lr: pd.Series(lr.predict(Xwhole),index=allt)).transpose()
predicts = predicts.applymap(np.exp)

predicts

fig,axes = plt.subplots(3,1,sharex=True)
monthly_costs = [29,49,99]
for index,cost in enumerate(monthly_costs):
    ax = axes[index]
    total_by_cost.loc[:,cost].plot(ax = ax,label='true values')
    predicts.loc[:,cost].plot(ax=ax,label='predictions')
    ax.legend(loc='best')
    ax.set_title('monthly cost = {}'.format(cost))
plt.rc('figure',figsize=(5,10))

pd.merge(total_by_cost,predicts,how='right',left_index=True,right_index=True,suffixes = ('_true','_pred'))

predicts.loc[12,:]/predicts.loc[1,:]

def calc_retention_rate(s):
    """
    input: 
        s: n-th value is #subscribers who paid 'n' cycles
    return:
        retention rate by the end of each cycle
    """
    r = s.iloc[::-1].cumsum().iloc[::-1]
    return r/r.iloc[0]

def retention_rate_by(colname):
    """
    step 1. group subscribers based on certain column, e.g., country or source
    step 2. for each group, count #subscribers who paid 'n' cycles
    step 3. for each group, calculate retention rate for each cycle
    """
    counts = subscriptions.groupby(colname).apply(lambda df: df.bill_cycles.value_counts()).unstack()
    return counts.apply(calc_retention_rate, axis=1).transpose()

retention_rate_by_country = retention_rate_by('country')
retention_rate_by_country

retention_rate_by_country.plot(marker='o')

# rank countries by August's retention rate
retention_rate_by_country.iloc[-1,:].sort_values(ascending=False)

retention_rate_by_source = retention_rate_by('source')
retention_rate_by_source

retention_rate_by_source.plot(marker='x')



