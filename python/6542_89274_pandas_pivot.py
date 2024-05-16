import pandas as pd
df = pd.read_csv("weather.csv")
df

df.pivot(index='city',columns='date')

df.pivot(index='city',columns='date',values="humidity")

df.pivot(index='date',columns='city')

df.pivot(index='humidity',columns='city')

df = pd.read_csv("weather2.csv")
df

df.pivot_table(index="city",columns="date")

df.pivot_table(index="city",columns="date", margins=True,aggfunc=np.sum)

df = pd.read_csv("weather3.csv")
df

df['date'] = pd.to_datetime(df['date'])

df.pivot_table(index=pd.Grouper(freq='M',key='date'),columns='city')

