def create_description_table(df, descriptions, round_num=2):
    df_desc = df.dtypes.to_frame(name='Data Type')
    df_desc['Description'] = descriptions
    df_desc['Missing Values'] = df.isnull().sum()
    df_desc['Mean'] = df.select_dtypes('number').mean().round(round_num)
    df_desc['Most Common'] = df.apply(lambda x: x.value_counts().index[0])
    df_desc['Most Common Ct'] = df.apply(lambda x: x.value_counts().iloc[0])
    df_desc['Unique Values'] = df.nunique()
    return df_desc

import pandas as pd

employee = pd.read_csv('../../data/employee.csv', parse_dates=['HIRE_DATE', 'JOB_DATE'])
employee.head()

employee.shape

descriptions = ['Position', 'Department', 'Base salary', 'Race', 
                'Full time/Part time/Temporary, etc...', 'Gender', 
                'Date hired', 'Date current job began']

create_description_table(employee, descriptions)

so = pd.read_csv('../../data/stackoverflow_qa.csv')
so.head()

so.shape

descriptions = ['Question ID', 'Creation date', '# of question upvotes', 'View count',
                'Question Title', 'Number of Answers', 'Number of comments for Question',
                'Number of favorites for Question', 'User name of question author',
                'Reputation of question author', 'User name of selected answer author',
                'Reputation of selected answer author']

create_description_table(so, descriptions)

food_inspections = pd.read_csv('../../data/food_inspections.csv', parse_dates=['Inspection Date'])
food_inspections.head()

food_inspections.shape

descriptions = ['Doing business as Name', 'Restaurant, Grocery store, School, Bakery, etc...',
                'High/Medium/Low', 'Address', 'Zip Code', 'Inspection Date',
                'Inspection Type', 'Pass/Fail/Out of business, etc...',
                'Detailed description of violations']

create_description_table(food_inspections, descriptions)



