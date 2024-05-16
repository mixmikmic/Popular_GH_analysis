import pandas as pd
df = pd.read_excel("stocks.xlsx",header=[0,1])
df

df.stack()

df.stack(level=0)

df_stacked=df.stack()
df_stacked

df_stacked.unstack()

df2 = pd.read_excel("stocks_3_levels.xlsx",header=[0,1,2])
df2

df2.stack()

df2.stack(level=0)

df2.stack(level=1)

