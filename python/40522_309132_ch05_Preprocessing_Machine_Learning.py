get_ipython().magic('reset -f')
from __future__ import division
import pandas as pd
import numpy as np

filename = '' #Write the filename of the original data set
df = pd.read_csv(filename, low_memory=False,skiprows=1)
col_names = df.columns.tolist()
print col_names
print 'Number of attributes: ' + str(len(col_names))

drop_cols = ['id', 'member_id', 'grade', 'sub_grade','earliest_cr_line', 'emp_title', 'verification_status', 'issue_d', 'loan_status', 'pymnt_plan', 'url', 'desc', 'purpose', 'title', 'zip_code', 'addr_state', 'inq_last_6mths', 'mths_since_last_record', 'open_acc', 'pub_rec', 'revol_bal', 'revol_util', 'initial_list_status', 'out_prncp', 'out_prncp_inv', 'total_pymnt', 'total_pymnt_inv', 'total_rec_prncp', 'total_rec_int', 'total_rec_late_fee', 'recoveries', 'collection_recovery_fee', 'last_pymnt_d', 'last_pymnt_amnt', 'next_pymnt_d', 'last_credit_pull_d', 'collections_12_mths_ex_med', 'mths_since_last_major_derog', 'policy_code']

df = df.drop(drop_cols,axis=1)

col_names = df.columns.tolist()
print col_names

get_ipython().magic('matplotlib inline')
loan = df['loan_amnt'].values
funded = df['funded_amnt_inv'].values
targets = np.abs(loan-funded)/loan

df['targets'] = targets
wrk_records = np.where(~np.isnan(targets))
y = targets[wrk_records]>=0.05

import matplotlib.pyplot as plt
plt.hist(targets[wrk_records],bins=30)

print 'Larger deviation: ' + str(np.sum(y))
print 'Total: ' + str(np.sum(1-y))

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
plt.pie(np.c_[len(y)-np.sum(y),np.sum(y)][0],labels=['Full amount','Not fully funded'],colors=['r','g'],shadow=True,autopct ='%.2f' )
fig = plt.gcf()
fig.set_size_inches(6,6)

df.head()

def clear_percent (row):
    try:
        d = float(row['int_rate'][:-1])/100.
    except:
        d = None
    return d

df['int_rate_clean'] = df.apply (lambda row: clear_percent(row),axis=1)
    

print 'Values of the variable term: ' + str(np.unique(df['term']))

def clear_term (row):
    try:
        if row['term']==' 36 months':
            d = 1
        else:
            if row['term']==' 60 months':
                d = 2
            else:
                if np.isnan(row['term']):
                    d = None
                else:
                    print 'WRONG'
                    print row['term']
    except:
        print 'EXCEPT'
        d = None
    return d

df['term_clean'] = df.apply (lambda row: clear_term(row),axis=1)
    

print 'Values for employment length: ' + str(np.unique(df['emp_length']))

#We use dictionary mapping as a switch 
def clean_emp_length(argument):
    switcher = {
        '1 year': 1,
        '2 years': 2,
        '3 years': 3,
        '4 years': 4,
        '5 years': 5,
        '6 years': 6,
        '7 years': 7,
        '8 years': 8,
        '9 years': 9,
        '10+ years': 10,
        '< 1 year': 0,
        'n/a':None,
    }
    try:
        d = switcher[argument['emp_length']]    
    except:
        d = None
    return d

df['emp_length_clean'] = df.apply (lambda row: clean_emp_length(row),axis=1)

df.head()

np.unique(df['home_ownership'])

from sklearn.feature_extraction import DictVectorizer

comb_dict = df[['home_ownership']].to_dict(orient='records')
vec = DictVectorizer()
home = 2*vec.fit_transform(comb_dict).toarray()-1
home[:5]

df_vector = pd.DataFrame(home[:,1:])
vector_columns = vec.get_feature_names()
df_vector.columns = vector_columns[1:]
df_vector.index = df.index
df_vector.head()

#Join data
df = df.join(df_vector)
df.head()

#Drop processed columns
df = df.drop(['term','int_rate','emp_length','home_ownership'],axis=1)
df.head()

#Drop the funded ammount
df=df.drop(['funded_amnt_inv'],axis=1)

#Declare targets
y = df['targets'].values>0.05
print 'Undefined values:' + str(np.sum(np.where(np.isnan(y),1,0)))
x=df.drop(['targets'],axis=1).values
idx_rmv = np.where(np.isnan(y))[0]
y = np.delete(y,idx_rmv)
x = np.delete(x,idx_rmv,axis=0)
print y.shape,x.shape

#Check what is going on in the data NaN
nan_feats=np.sum(np.where(np.isnan(x),1,0),axis=0)
plt.bar(np.arange(len(nan_feats)),nan_feats)
fig = plt.gcf()
fig.set_size_inches((12,5))
nan_feats

#Drop feature 6, too much NaN
print col_names[6]
x=np.hstack((x[:,:6],x[:,7:]))

x.shape
#Check now
nan_feats=np.sum(np.where(np.isnan(x),1,0),axis=0)
plt.bar(np.arange(len(nan_feats)),nan_feats)
fig = plt.gcf()
fig.set_size_inches((12,5))
nan_feats

#Check records
nan_records=np.sum(np.where(np.isnan(x),1,0),axis=1)
np.histogram(nan_records)

print len(nan_records),len(y)
idx_rmv = np.where(nan_records>0)[0]
y = np.delete(y,idx_rmv)
x = np.delete(x,idx_rmv,axis=0)
print y.shape,x.shape



