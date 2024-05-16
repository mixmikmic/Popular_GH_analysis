get_ipython().magic('matplotlib inline')

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("movies_genres.csv", delimiter='\t')
df.info()

df_genres = df.drop(['plot', 'title'], axis=1)
counts = []
categories = list(df_genres.columns.values)
for i in categories:
    counts.append((i, df_genres[i].sum()))
df_stats = pd.DataFrame(counts, columns=['genre', '#movies'])
df_stats

df_stats.plot(x='genre', y='#movies', kind='bar', legend=False, grid=True, figsize=(15, 8))

df.drop('Lifestyle', axis=1, inplace=True)

from langdetect import detect
df['plot_lang'] = df.apply(lambda row: detect(row['plot'].decode("utf8")), axis=1)
df['plot_lang'].value_counts()

df = df[df.plot_lang.isin(['en'])]
df.to_csv("movies_genres_en.csv", sep='\t', encoding='utf-8')

