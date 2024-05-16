from dask import array, dataframe, bag

da = array.random.binomial(100, .3, 1000, chunks=(100))

da.size

da[:2]

da[:2].compute()

da.npartitions

da[:400].chunks

da.min().visualize()

da.min().compute()

df = dataframe.read_csv('../data/speakers.csv')

df.head()

df.dtypes

df.npartitions

df.twitter.loc[4]

df.twitter.loc[4].compute()

df.groupby(['nationality', 'curr_pol_group_abbr'])

df.groupby(['nationality', 'curr_pol_group_abbr']).sum().visualize()

df2 = df.set_index('speaker_id')

df2.npartitions

df.speaker_id.compute().values

partitions = sorted(df.speaker_id.compute().values)[::50]

partitions

df3 = df.set_partition('speaker_id', partitions)

df3.npartitions

df3.divisions

df3.groupby(['nationality', 'curr_pol_group_abbr']).sum().visualize()

df3.groupby(['nationality', 'curr_pol_group_abbr']).sum().compute()

db = bag.read_text('../data/europarl_speech_text.txt')

db.take(2)

db.filter(lambda x: 'Czech' in x)

db.filter(lambda x: 'Czech' in x).take(3)

db.filter(lambda x: 'Czech' in x).str.split().concat().frequencies().topk(100)

db.filter(lambda x: 'Czech' in x).str.split().concat().frequencies().topk(100).compute()



