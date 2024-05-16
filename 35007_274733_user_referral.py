import datetime
import pandas as pd
import scipy.stats as ss
import matplotlib.pyplot as plt
plt.style.use('ggplot')

referral = pd.read_csv("referral.csv")
del referral['device_id']
referral['date'] = pd.to_datetime( referral.date )

referral.head()# glance the data

dt_referral_starts = datetime.datetime(2015,10,31)

referral.date.describe()

(pd.Series(referral.date.unique()) >= dt_referral_starts).value_counts()

def count_spent(df):
    d = {}
    d['n_purchase'] = df.shape[0]# number of purchase in that day
    d['total_spent'] = df.money_spent.sum() # total money spent in that day
    d['n_customer'] = df.user_id.unique().shape[0] # how many customers access the store that day
    return pd.Series(d)

def daily_statistics(df):
    """
    given a dataframe
    1.  group by day, and return '#purchase','total spent money','#customers' on each day
    2.  split daily data into two groups, before the program and after the program
    3.  for each 'sale index' ('#purchase','total spent money','#customers'), 
        calculate the mean before/after the program, their difference, and pvalue 
    """
    grpby_day = df.groupby('date').apply(count_spent)

    grpby_day_before = grpby_day.loc[grpby_day.index < dt_referral_starts, :]
    grpby_day_after = grpby_day.loc[grpby_day.index >= dt_referral_starts, :]

    d = []
    colnames = ['total_spent','n_purchase','n_customer']
    for col in colnames:
        pre_data = grpby_day_before.loc[:,col]
        pre_mean = pre_data.mean()

        post_data = grpby_day_after.loc[:,col]
        post_mean = post_data.mean()

        result = ss.ttest_ind(pre_data, post_data, equal_var=False)
        # either greater or smaller, just one-tail test
        pvalue = result.pvalue / 2 

        d.append({'mean_pre':pre_mean,'mean_post':post_mean,'mean_diff':post_mean - pre_mean,
                  'pvalue':pvalue})

    # re-order the columns
    return pd.DataFrame(d,index = colnames).loc[:,['mean_pre','mean_post','mean_diff','pvalue']]

daily_statistics(referral)

referral.country.value_counts()

daily_stat_bycountry = referral.groupby('country').apply(daily_statistics)

daily_stat_bycountry

daily_stat_bycountry.xs('total_spent',level=1).sort_values(by='pvalue')

daily_stat_bycountry.xs('n_customer',level=1).sort_values(by='pvalue')

daily_stat_bycountry.xs('n_purchase',level=1).sort_values(by='pvalue')

