import pandas as pd
df = pd.read_csv("weather.csv")
df

melted = pd.melt(df, id_vars=["day"], var_name='city', value_name='temperature')
melted

