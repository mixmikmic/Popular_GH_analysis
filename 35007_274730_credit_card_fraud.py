import itertools
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.cluster import KMeans

get_ipython().magic('matplotlib inline')
sns.set_context('notebook')

holders = pd.read_csv('cc_info.csv',index_col='credit_card')
holders.rename(columns={'credit_card_limit':'credit_limit'},inplace=True)

transactions = pd.read_csv('transactions.csv')
transactions['date'] = pd.to_datetime(transactions.date)
transactions.rename(columns={'transaction_dollar_amount':'amount'},inplace=True)

transactions.date.dt.year.value_counts()

def monthly_spent_byuser(df):
    # I have checked the data already, all transactions happen in year 2015
    # so I can just group by month
    return df.groupby(df.date.dt.month)['amount'].agg('sum')

# first group by 'credit_card' (i.e., by user)
# then sum up all spent by month
card_month_spents = transactions.groupby("credit_card").apply(monthly_spent_byuser).unstack(fill_value=0)

# join with 'credit_limit' to simplify the comparison
card_month_spents = card_month_spents.join(holders.credit_limit)
card_month_spents.head()

n_months = card_month_spents.shape[1]-1
def is_never_above_limit(s):
    limit = s.loc['credit_limit']
    return (s.iloc[0:n_months] <= limit).all()

is_user_never_exceed_limit = card_month_spents.apply(is_never_above_limit,axis=1)

users_never_exceed_limit = card_month_spents.loc[is_user_never_exceed_limit ,:].index

users_never_exceed_limit

with open("users_never_exceed_limit.txt","wt") as outf:
    for cardno in users_never_exceed_limit:
        outf.write('{}\n'.format(cardno))

class MonthSpentMonitor(object):

    def __init__(self,credit_limits):
        """
        card_limits is a dictionary
        key=card number, value=credit limit
        """
        self.total_spent = defaultdict(float)
        self.credit_limits = credit_limits

    def reset(self):
        self.total_spent.clear()

    def count(self,daily_transaction):
        """
        daily_transaction: a dict
        key=card number, value=amount
        """
        for cardno,amount in daily_transaction:
            self.total_spent[cardno] += amount

        # assume 'credit_limits' always can find the cardno
        # otherwise, raise KeyError, which is a good indicator showing something is wrong
        return [ cardno for cardno,total in self.total_spent.viewitems() if total > self.credit_limits[cardno]]

def statistics_by_card(s):
    ps = [25, 50, 75]
    d = np.percentile(s,ps)
    return pd.Series(d,index=['{}%'.format(p) for p in ps])

tran_statistics = transactions.groupby('credit_card')['amount'].apply(statistics_by_card).unstack()

tran_statistics.head()

# merge 'transaction' with 'previous consumption statistics'
temp = pd.merge(transactions,tran_statistics,how='left',left_on='credit_card',right_index=True)

# merge with credit limit
transactions = pd.merge(temp,holders.loc[:,['credit_limit']],how='left',left_on='credit_card',right_index=True)

transactions.tail()

# save it for later use
transactions.to_csv('extend_transactions.csv',index=False)

X = transactions.loc[:,['amount','25%','50%','75%','credit_limit']]

X.describe()

X = scale(X)

pca = PCA(n_components=2)
X2d = pca.fit_transform(X)
X2d = pd.DataFrame(X2d,columns=['pc1','pc2'])

plt.scatter(X2d.pc1,X2d.pc2,alpha=0.3)

n_clusters = 6
kmeans = KMeans(n_clusters=n_clusters,n_jobs=-1)
kmeans.fit(X)

X2d['label'] = kmeans.labels_
print X2d.label.value_counts()

colors = itertools.cycle( ['r','g','b','c','m','y','k'] )

plt.rc('figure',figsize=(10,6))
for label in  xrange(n_clusters) :
    temp = X2d.loc[X2d.label == label,:]
    plt.scatter(temp.pc1,temp.pc2,c=next(colors),label=label,alpha=0.3)

plt.legend(loc='best')

X2d.head()

g = sns.FacetGrid(X2d, hue="label")
g.map(plt.scatter, "pc1", "pc2", alpha=0.3)
g.add_legend();

suspicious_label = X2d.label.value_counts().argmin()
suspicious_label

suspect = transactions.loc[X2d.label==suspicious_label,['credit_card','amount','25%','50%','75%','credit_limit','date']]
suspect.to_csv('suspect.csv',index=False)

suspect.sample(10)

labels = ["amount",'75%']
plt.hist(suspect.loc[:,labels].values,bins=50,label=labels)
plt.legend(loc='best')



