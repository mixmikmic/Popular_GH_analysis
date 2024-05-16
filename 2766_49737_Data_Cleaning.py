import numpy as np
import pandas as pd

get_ipython().system('pwd')

# The cleaned data file is saved here:
output_file = "../data/coal_prod_cleaned.csv"

df7 = pd.read_csv("../data/coal_prod_2008.csv", index_col="MSHA_ID")
df8 = pd.read_csv("../data/coal_prod_2009.csv", index_col="MSHA_ID")
df9 = pd.read_csv("../data/coal_prod_2010.csv", index_col="MSHA_ID")
df10 = pd.read_csv("../data/coal_prod_2011.csv", index_col="MSHA_ID")
df11 = pd.read_csv("../data/coal_prod_2012.csv", index_col="MSHA_ID")

dframe = pd.concat((df7, df8, df9, df10, df11))

# Noticed a probable typo in the data set: 
dframe['Company_Type'].unique()

# Correcting the Company_Type
dframe.loc[dframe['Company_Type'] == 'Indepedent Producer Operator', 'Company_Type'] = 'Independent Producer Operator'
dframe.head()

dframe[dframe.Year == 2008].head()

dframe.to_csv(output_file, )



