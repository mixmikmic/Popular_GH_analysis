import pandas as pd
import numpy as np

college = pd.read_csv('data/college.csv')
columns = college.columns
columns

columns.values

columns[5]

columns[[1,8,10]]

columns[-7:-4]

columns.min(), columns.max(), columns.isnull().sum()

columns + '_A'

columns > 'G'

columns[1] = 'city'

c1 = columns[:4]
c1

c2 = columns[2:5]
c2

c1.union(c2)

c1 | c2

c1.symmetric_difference(c2)

c1 ^ c2

s1 = pd.Series(index=list('aaab'), data=np.arange(4))
s1

s2 = pd.Series(index=list('cababb'), data=np.arange(6))
s2

s1 + s2

s1 = pd.Series(index=list('aaabb'), data=np.arange(5))
s2 = pd.Series(index=list('aaabb'), data=np.arange(5))
s1 + s2

s1 = pd.Series(index=list('aaabb'), data=np.arange(5))
s2 = pd.Series(index=list('bbaaa'), data=np.arange(5))
s1 + s2

employee = pd.read_csv('data/employee.csv', index_col='RACE')
employee.head()

salary1 = employee['BASE_SALARY']
salary2 = employee['BASE_SALARY']
salary1 is salary2

salary1 = employee['BASE_SALARY'].copy()
salary2 = employee['BASE_SALARY'].copy()
salary1 is salary2

salary1 = salary1.sort_index()
salary1.head()

salary2.head()

salary_add = salary1 + salary2

salary_add.head()

salary_add1 = salary1 + salary1
len(salary1), len(salary2), len(salary_add), len(salary_add1)

index_vc = salary1.index.value_counts(dropna=False)
index_vc

index_vc.pow(2).sum()

baseball_14 = pd.read_csv('data/baseball14.csv', index_col='playerID')
baseball_15 = pd.read_csv('data/baseball15.csv', index_col='playerID')
baseball_16 = pd.read_csv('data/baseball16.csv', index_col='playerID')
baseball_14.head()

baseball_14.index.difference(baseball_15.index)

baseball_14.index.difference(baseball_15.index)

hits_14 = baseball_14['H']
hits_15 = baseball_15['H']
hits_16 = baseball_16['H']
hits_14.head()

(hits_14 + hits_15).head()

hits_14.add(hits_15, fill_value=0).head()

hits_total = hits_14.add(hits_15, fill_value=0).add(hits_16, fill_value=0)
hits_total.head()

hits_total.hasnans

s = pd.Series(index=['a', 'b', 'c', 'd'], data=[np.nan, 3, np.nan, 1])
s

s1 = pd.Series(index=['a', 'b', 'c'], data=[np.nan, 6, 10])
s1

s.add(s1, fill_value=5)

s1.add(s, fill_value=5)

df_14 = baseball_14[['G','AB', 'R', 'H']]
df_14.head()

df_15 = baseball_15[['AB', 'R', 'H', 'HR']]
df_15.head()

(df_14 + df_15).head(10).style.highlight_null('yellow')

df_14.add(df_15, fill_value=0).head(10).style.highlight_null('yellow')

employee = pd.read_csv('data/employee.csv')
dept_sal = employee[['DEPARTMENT', 'BASE_SALARY']]

dept_sal = dept_sal.sort_values(['DEPARTMENT', 'BASE_SALARY'],
                                ascending=[True, False])

max_dept_sal = dept_sal.drop_duplicates(subset='DEPARTMENT')
max_dept_sal.head()

max_dept_sal = max_dept_sal.set_index('DEPARTMENT')
employee = employee.set_index('DEPARTMENT')

employee['MAX_DEPT_SALARY'] = max_dept_sal['BASE_SALARY']

pd.options.display.max_columns = 6

employee.head()

employee.query('BASE_SALARY > MAX_DEPT_SALARY')

np.random.seed(1234)
random_salary = dept_sal.sample(n=10).set_index('DEPARTMENT')
random_salary

employee['RANDOM_SALARY'] = random_salary['BASE_SALARY']

employee['MAX_SALARY2'] = max_dept_sal['BASE_SALARY'].head(3)

employee.MAX_SALARY2.value_counts()

employee.MAX_SALARY2.isnull().mean()

pd.options.display.max_rows = 8

college = pd.read_csv('data/college.csv', index_col='INSTNM')
college.dtypes

college.MD_EARN_WNE_P10.iloc[0]

college.GRAD_DEBT_MDN_SUPP.iloc[0]

college.MD_EARN_WNE_P10.sort_values(ascending=False).head()

cols = ['MD_EARN_WNE_P10', 'GRAD_DEBT_MDN_SUPP']
for col in cols:
    college[col] = pd.to_numeric(college[col], errors='coerce')

college.dtypes.loc[cols]

college_n = college.select_dtypes(include=[np.number])
college_n.head() # only numeric columns

criteria = college_n.nunique() == 2
criteria.head()

binary_cols = college_n.columns[criteria].tolist()
binary_cols

college_n2 = college_n.drop(labels=binary_cols, axis='columns')
college_n2.head()

max_cols = college_n2.idxmax()
max_cols

unique_max_cols = max_cols.unique()
unique_max_cols[:5]

college_n2.loc[unique_max_cols].style.highlight_max()

college = pd.read_csv('data/college.csv', index_col='INSTNM')
college_ugds = college.filter(like='UGDS_').head()
college_ugds.style.highlight_max(axis='columns')

pd.Timedelta(1, unit='Y')

college = pd.read_csv('data/college.csv', index_col='INSTNM')

cols = ['MD_EARN_WNE_P10', 'GRAD_DEBT_MDN_SUPP']
for col in cols:
    college[col] = pd.to_numeric(college[col], errors='coerce')

college_n = college.select_dtypes(include=[np.number])
criteria = college_n.nunique() == 2
binary_cols = college_n.columns[criteria].tolist()
college_n = college_n.drop(labels=binary_cols, axis='columns')

college_n.max().head()

college_n.eq(college_n.max()).head()

has_row_max = college_n.eq(college_n.max()).any(axis='columns')
has_row_max.head()

college_n.shape

has_row_max.sum()

pd.options.display.max_rows=6

college_n.eq(college_n.max()).cumsum().cumsum()

has_row_max2 = college_n.eq(college_n.max())                        .cumsum()                        .cumsum()                        .eq(1)                        .any(axis='columns')
has_row_max2.head()

has_row_max2.sum()

idxmax_cols = has_row_max2[has_row_max2].index
idxmax_cols

set(college_n.idxmax().unique()) == set(idxmax_cols)

get_ipython().run_line_magic('timeit', 'college_n.idxmax().values')

get_ipython().run_line_magic('timeit', "college_n.eq(college_n.max())                              .cumsum()                              .cumsum()                              .eq(1)                              .any(axis='columns')                              [lambda x: x].index")

pd.options.display.max_rows= 40

college = pd.read_csv('data/college.csv', index_col='INSTNM')
college_ugds = college.filter(like='UGDS_')
college_ugds.head()

highest_percentage_race = college_ugds.idxmax(axis='columns')
highest_percentage_race.head()

highest_percentage_race.value_counts(normalize=True)

college_black = college_ugds[highest_percentage_race == 'UGDS_BLACK']
college_black = college_black.drop('UGDS_BLACK', axis='columns')
college_black.idxmax(axis='columns').value_counts(normalize=True)



