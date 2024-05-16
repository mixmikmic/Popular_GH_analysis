import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier,export_graphviz
from sklearn.feature_selection import chi2
import matplotlib.pyplot as plt
plt.style.use("ggplot")
get_ipython().magic('matplotlib inline')

allusers = pd.read_csv("home_page_table.csv",index_col="user_id")
users_to_search = pd.read_csv("search_page_table.csv",index_col="user_id")
users_to_pay = pd.read_csv("payment_page_table.csv",index_col="user_id")
users_to_confirm = pd.read_csv("payment_confirmation_table.csv",index_col="user_id")

allusers.loc[users_to_search.index,"page"] = users_to_search.page
allusers.loc[users_to_pay.index,"page"] = users_to_pay.page
allusers.loc[users_to_confirm.index,"page"] = users_to_confirm.page

# give it a better, more clear name
allusers.rename(columns={'page':'final_page'},inplace=True)

# change string to ordered-categorical feature
pages = ["home_page","search_page","payment_page","payment_confirmation_page"]
allusers["final_page"] = allusers.final_page.astype("category",categories = pages,ordered=True)

user_infos = pd.read_csv("user_table.csv",index_col="user_id")
user_infos.loc[:,"date"] = pd.to_datetime(user_infos.date)

allusers = allusers.join(user_infos)
allusers.head()

allusers.to_csv("all_users.csv",index_label="user_id")

def conversion_rates(df):
    stage_counts = df.final_page.value_counts()
    # #users converts from current page
    convert_from = stage_counts.copy()

    total = df.shape[0]
    for page in stage_counts.index:
        n_left = stage_counts.loc[page]# how many users just stop at current page
        n_convert = total - n_left
        convert_from[page] = n_convert
        total = n_convert

    cr = pd.concat([stage_counts,convert_from],axis=1,keys=["n_drop","n_convert"])
    cr["convert_rates"] = cr.n_convert.astype(np.float)/(cr.n_drop + cr.n_convert)
    cr['drop_rates'] = 1 - cr.convert_rates

    return cr

allusers.groupby('device').apply(conversion_rates)

allusers.groupby('device')['final_page'].apply(lambda s: s.value_counts(normalize=True)).unstack()

allusers.head()

X = allusers.copy()

X.device.value_counts()

X['from_mobile'] = (X.device == 'Mobile').astype(int)
del X['device']

X['is_male'] = (X.sex == 'Male').astype(int)
del X['sex']

X['converted'] = (X.final_page == 'payment_confirmation_page').astype(int)
del X['final_page']

X.converted.mean()# a highly imbalanced classification problem

X.date.describe()

X['weekday'] = X.date.dt.weekday_name
del X['date']

X.head()

X.groupby('weekday')['converted'].agg(['count','mean']).sort_values(by='mean',ascending=False)

X.groupby('is_male')['converted'].agg(['count','mean']).sort_values(by='mean',ascending=False)

X = pd.get_dummies(X,prefix='',prefix_sep='')
X.head()

y = X.converted
X = X.loc[:,X.columns != 'converted']

scores, pvalues = chi2(X,y)

pd.DataFrame({'chi2_score':scores,'chi2_pvalue':pvalues},index=X.columns).sort_values(by='chi2_score',ascending=False)







del X['Tuesday']# remove one redundant feature



dt = DecisionTreeClassifier(max_depth=3,min_samples_leaf=20,min_samples_split=20)
dt.fit(X,y)
export_graphviz(dt,feature_names=X.columns,class_names=['NotConvert','Converted'],
                proportion=True,leaves_parallel=True,filled=True)



