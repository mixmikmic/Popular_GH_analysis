import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('graduates.csv')
df.head()

new_majors = df['Major'][df['Asians'] == 0].unique()
new_majors
df = df[~df.Major.isin(new_majors)]
# df.nunique()

major_df = df.groupby(['Major']).sum().reset_index()
# major_df.head().T

plt.figure(figsize=(20,10))
g = sns.boxplot(x="Major", y="Total",data=df)

g.set_xticklabels(g.get_xticklabels(),rotation=20)
plt.title('Major v/s Number of Graduates')
plt.show() 

plt.figure(figsize=(20,10))
g = sns.factorplot(x="Major", y="Unemployed", data=df, size=12, kind="bar",hue = 'Year')
g.set_xticklabels(rotation=20)
plt.title('Number of unemployed graduates per major')
plt.show()

df['Employment Rate'] = df['Employed']/df['Total']
df.head()

plt.figure(figsize=(20,10))
g = sns.stripplot(x="Major", y="Employment Rate",hue = 'Year', data=df, size = 15)
g.set_xticklabels(g.get_xticklabels(),rotation=20)
plt.legend(loc = 'lower right')
plt.title('Emploment Rate for every major over the years')
plt.show()

f = lambda x: 'new' if x.Year > 2003 else 'old'
dfc = df.copy()
dfc['old_new'] = dfc.apply(f, axis=1)

plt.figure(figsize=(20,10))
g = sns.violinplot(x="old_new", y="Employment Rate", data=dfc, hue = 'Major');
plt.legend(loc = 'lower left')
plt.title('Distribution of Employment Rate pre and post 2003')
plt.xlabel('Pre and Post 2003')
plt.show()

df['Gender Ratio'] = df['Males']/df['Females']
major_df['Gender Ratio'] = major_df['Males']/major_df['Females']

plt.figure(figsize=(20,10))
g = sns.stripplot(x="Major", y='Gender Ratio', data=df,color = 'red',  size = 10, jitter = True)
g.set_xticklabels(g.get_xticklabels(),rotation=20)
plt.title('Major v/s Gender Ratio')
plt.show()

df1 = df[['Major', 'Year', 'Males', 'Females']]
df1 = pd.melt(df1, ['Major', 'Year'], var_name="Gender")

plt.figure(figsize=(20,10))
sns.lmplot(x="Year", y="value", col="Gender", hue="Major", data=df1,
           col_wrap=2, ci=None, palette="husl", size=6, 
           scatter_kws={"s": 50, "alpha": 1})
plt.show()

plt.figure(figsize = (15,8))
g = sns.swarmplot(x="Year", y="Asians", data=df, size = 8, hue = 'Major')
g.set_xticklabels(g.get_xticklabels(),rotation=20)
plt.title('Number of Asians over the Years')
plt.show()

plt.figure(figsize = (15,8))
g = sns.swarmplot(x="Year", y="Minorities", data=df, size = 8, hue = 'Major')
plt.title('Number of Minorities over the Years')
plt.show()

df1 = df[['Major', 'Year', 'Whites', 'Asians', 'Minorities']]
df1 = pd.melt(df1, ['Major', 'Year'], var_name="Race")

plt.figure(figsize=(10,20))
g = sns.factorplot(x="Year", y="value", hue="Race", data=df1, 
                   size=10, kind="bar", palette="muted", legend_out="True") # legend_out draws the legend outside the chart

g.set_ylabels("Distribution by race")
plt.title('Distribution of Race by Year')
plt.show()

# Binning the data based on years into 2 i.e. before and after 2001
df.columns
df.Major.unique()
df.Year.unique()

f = lambda x: 'new' if x.Year > 2003 else 'old'
dfc = df.copy()
dfc['old_new'] = dfc.apply(f, axis=1)

g = sns.FacetGrid(dfc, col="old_new", row = 'Major', margin_titles=True)
g.map(plt.hist, "Doctorates", color="steelblue",bins = 3)
plt.show() 

