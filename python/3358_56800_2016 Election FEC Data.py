import pandas as pd

import numpy as np

from csv import reader

cand_header = [r for r in reader(open('data/cn_header_file.csv', 'r'))]

cand_header

candidates = pd.read_csv('data/cn.txt', names=cand_header[0], sep='|')

candidates.head()

candidates['CAND_NAME']

candidates['CAND_ELECTION_YR'] == 2016

candidates[candidates['CAND_ELECTION_YR'] == 2016]

candidates[candidates['CAND_ELECTION_YR'] == 2016][['CAND_ID', 'CAND_NAME']].head()

candidates.shape

candidates.loc[6940:]

candidates.iloc[2]

candidates.dtypes

candidates[candidates['CAND_NAME'] == 'TRUMP, DONALD']

candidates[candidates['CAND_NAME'].str.contains('TRUMP')]

candidates[candidates['CAND_NAME'].notnull() & candidates['CAND_NAME'].str.contains('TRUMP')]

donations_header = [r for r in reader(open('data/indiv_header_file.csv', 'r'))]

donations_header[0]

donations = pd.read_csv('data/itcont.txt', names=donations_header[0], sep='|')

donations.head()

donations.dtypes

donations.describe()

donations['TRANSACTION_AMT'].mean()

donations['TRANSACTION_AMT'].min()

donations['TRANSACTION_AMT'].max()

donations['TRANSACTION_AMT'].median()

get_ipython().magic('pylab inline')

hist(donations['TRANSACTION_AMT'])

candidates[candidates['CAND_PCC'].notnull()].shape

donations.set_index('CMTE_ID').join(candidates.set_index('CAND_PCC'))

donations.set_index('CMTE_ID').join(candidates.set_index('CAND_PCC'), how='right')

donations.set_index('CMTE_ID').join(candidates.set_index('CAND_PCC'), how='inner')

cand_donations = donations.set_index('CMTE_ID').join(candidates.set_index('CAND_PCC'), how='inner')

cand_donations.describe()

hist(cand_donations['TRANSACTION_AMT'])

cand_donations['TRANSACTION_AMT'].max()

cand_donations[cand_donations['TRANSACTION_AMT'] > 1000000]

cand_donations[cand_donations['TRANSACTION_AMT'] > 1000000]['CAND_NAME'].value_counts()

cand_donations[cand_donations['TRANSACTION_AMT'] < 200]['CAND_NAME'].value_counts()

cand_donations.columns

cand_donations = cand_donations[cand_donations['CAND_ELECTION_YR'] == 2016]

grouped = cand_donations.groupby('CAND_NAME')

grouped.sum()

grouped.agg({'TRANSACTION_AMT': [np.sum, np.mean], 'NAME': lambda x: len(set(x))})

cand_donations['unique_donors'] = cand_donations.groupby('CAND_NAME')['NAME'].transform(lambda x: 
                                                                                        len(set(x)))

cand_donations['unique_donors'].mean()

cand_donations['unique_donors'].median()

sign_cand_donations = cand_donations[cand_donations['unique_donors'] > cand_donations['unique_donors'].mean()]

sign_cand_donations.shape

sign_cand_donations.groupby('CAND_NAME').sum()

cand_donations[cand_donations['CAND_NAME'].str.contains('TRUMP')]['unique_donors']

cand_donations[cand_donations['CAND_NAME'].str.contains('TRUMP')].describe()

sign_cand_donations = sign_cand_donations.append(cand_donations[cand_donations['CAND_NAME'].str.contains('TRUMP')])

sign_cand_donations.groupby('CAND_NAME').sum()['TRANSACTION_AMT']

sign_cand_donations.groupby('CAND_NAME').min()['unique_donors'].sort_values()



comm_header = [r for r in reader(open('data/cm_header_file.csv', 'r'))]

committees = pd.read_csv('data/cm.txt', names=comm_header[0], sep='|')

committees.head()



