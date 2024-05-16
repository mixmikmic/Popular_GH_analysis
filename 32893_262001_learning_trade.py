import zipfile
s_fname = "data/data_0725_0926.zip"
s_fname2 = "data/bova11_2.zip"
archive = zipfile.ZipFile(s_fname, 'r')
archive2 = zipfile.ZipFile(s_fname2, 'r')
l_fnames = archive.infolist()

import qtrader.eda as eda; reload(eda);
df_last_pnl = eda.plot_cents_changed(archive, archive2)

def foo():
    f_total = 0.
    f_tot_rows = 0.
    for i, x in enumerate(archive.infolist()):
        f_total += x.file_size/ 1024.**2
        for num_rows, row in enumerate(archive.open(x)):
            f_tot_rows += 1
        print "{}:\t{:,.0f} rows\t{:0.2f} MB".format(x.filename, num_rows + 1, x.file_size/ 1024.**2)
    print '=' * 42
    print "TOTAL\t\t{} files\t{:0.2f} MB".format(i+1,f_total)
    print "\t\t{:0,.0f} rows".format(f_tot_rows)

get_ipython().magic('time foo()')

import pandas as pd
df = pd.read_csv(archive.open(l_fnames[0]), index_col=0, parse_dates=['Date'])
df.head()

import qtrader.simulator as simulator
import qtrader.environment as environment
e = environment.Environment()
sim = simulator.Simulator(e)
get_ipython().magic('time sim.run(n_trials=1)')

sim.env.get_order_book()

import qtrader.eda as eda; reload(eda);
s_fname = "data/petr4_0725_0818_2.zip"
get_ipython().magic('time eda.test_ofi_indicator(s_fname, f_min_time=20.)')

import pandas as pd
df = pd.read_csv('data/ofi_petr.txt', sep='\t')
df.drop('TIME', axis=1, inplace=True)
df.dropna(inplace=True)
ax = sns.lmplot(x="OFI", y="LOG_RET", data=df, markers=["x"], palette="Set2", size=4, aspect=2.)
ax.ax.set_title(u'Relation between the Log-return and the $OFI$\n', fontsize=15);
ax.ax.set_ylim([-0.004, 0.005])
ax.ax.set_xlim([-400000, 400000])

import pandas as pd
df = pd.read_csv('data/ofi_petr.txt', sep='\t')
df.drop(['TIME', 'DELTA_MID'], axis=1, inplace=True)
df.dropna(inplace=True)

# Produce a scatter matrix for each pair of features in the data
pd.scatter_matrix(df.ix[:, ['OFI', 'BOOK_RATIO']],
                  alpha = 0.3, figsize = (14,8), diagonal = 'kde');

import sklearn.preprocessing as preprocessing
import numpy as np

scaler_ofi = preprocessing.MinMaxScaler().fit(pd.DataFrame(df.OFI))
scaler_bookratio = preprocessing.MinMaxScaler().fit(pd.DataFrame(np.log(df.BOOK_RATIO)))
d_transformed = {}
d_transformed['OFI'] = scaler_ofi.transform(pd.DataFrame(df.OFI)).T[0]
d_transformed['BOOK_RATIO'] = scaler_bookratio.transform(pd.DataFrame(np.log(df.BOOK_RATIO))).T[0]

df_transformed = pd.DataFrame(d_transformed)
pd.scatter_matrix(df_transformed.ix[:, ['OFI', 'BOOK_RATIO']],
                    alpha = 0.3, figsize = (14,8), diagonal = 'kde');

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.mixture import GMM
import time


reduced_data = df_transformed.ix[:, ['OFI', 'BOOK_RATIO']]
reduced_data.columns = ['Dimension 1', 'Dimension 2']
range_n_clusters = [2, 3, 4, 5, 6, 8, 10]

f_st = time.time()
d_score = {}
d_model = {}
s_key = "Kmeans"
d_score[s_key] = {}
d_model[s_key] = {}
for n_clusters in range_n_clusters:
    # TODO: Apply your clustering algorithm of choice to the reduced data 
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    preds = clusterer.fit_predict(reduced_data)
    d_model[s_key][n_clusters] = clusterer
    d_score[s_key][n_clusters] = metrics.silhouette_score(reduced_data, preds)
print "K-Means took {:0.2f} seconds to run over all complexity space".format(time.time() - f_st)

f_avg = 0

for covar_type in ['spherical', 'diag', 'tied', 'full']:
    f_st = time.time()
    s_key = "GMM_{}".format(covar_type)
    d_score[s_key] = {}
    d_model[s_key] = {}
    for n_clusters in range_n_clusters:
        
        # TODO: Apply your clustering algorithm of choice to the reduced data 
        clusterer = GMM(n_components=n_clusters,
                        covariance_type=covar_type,
                        random_state=10)
        clusterer.fit(reduced_data)
        preds = clusterer.predict(reduced_data)
        d_model[s_key][n_clusters] = clusterer
        d_score[s_key][n_clusters] = metrics.silhouette_score(reduced_data, preds)
        f_avg += time.time() - f_st
        
print "GMM took {:0.2f} seconds on average to run over all complexity space".format(f_avg / 4.)

import pandas as pd
ax = pd.DataFrame(d_score).plot()
ax.set_xlabel("Number of Clusters")
ax.set_ylabel("Silhouette Score\n")
ax.set_title("Performance vs Complexity\n", fontsize = 16);

# get centers
sample_preds = []
centers = d_model["Kmeans"][6].cluster_centers_
preds = d_model["Kmeans"][6].fit_predict(reduced_data)

# Display the results of the clustering from implementation
import qtrader.eda as eda; reload(eda);
eda.cluster_results(reduced_data, preds, centers)

# recovering data
log_centers = centers.copy()
df_aux = pd.DataFrame([np.exp(scaler_bookratio.inverse_transform(log_centers.T[0].reshape(1, -1))[0]),
                      scaler_ofi.inverse_transform(log_centers.T[1].reshape(1, -1))[0]]).T
df_aux.columns = df_transformed.columns
df_aux.index.name = 'CLUSTER'
df_aux.columns = ['BOOK RATIO', 'OFI']
df_aux.round(2)

import pickle
pickle.dump(d_model["Kmeans"][6] ,open('data/kmeans_2.dat', 'w'))
pickle.dump(scaler_ofi, open('data/scale_ofi_2.dat', 'w'))
pickle.dump(scaler_bookratio, open('data/scale_bookratio_2.dat', 'w'))
print 'Done !'

# analyze the logs from the in-sample tests
import qtrader.eda as eda;reload(eda);
s_fname = 'log/train_test/sim_Fri_Oct__7_002946_2016.log'  # 15 old
# s_fname = 'log/train_test/sim_Wed_Oct__5_110344_2016.log'  # 15
# s_fname = 'log/train_test/sim_Thu_Oct__6_165539_2016.log'  # 25
# s_fname = 'log/train_test/sim_Thu_Oct__6_175507_2016.log'  # 35
# s_fname = 'log/train_test/sim_Thu_Oct__6_183555_2016.log'  # 5
get_ipython().magic("time d_rtn_train_1 = eda.simple_counts(s_fname, 'LearningAgent_k')")

import qtrader.eda as eda; reload(eda);
eda.plot_train_test_sim(d_rtn_train_1)

# improving K
import qtrader.eda as eda;reload(eda);
s_fname = 'log/train_test/sim_Thu_Oct__6_133518_2016.log'
get_ipython().magic("time d_rtn_k = eda.count_by_k_gamma(s_fname, 'LearningAgent_k', 'k')")

import pandas as pd
import matplotlib.pyplot as plt

f, na_ax = plt.subplots(1, 4, sharex=True, sharey=True)
for ax1, s_key in zip(na_ax.ravel(), ['0.3', '0.8', '1.3', '2.0']):
    df_aux = pd.Series(d_rtn_k[s_key][5])
    df_filter = pd.Series([x.hour for x in df_aux.index])
    df_aux = df_aux[((df_filter < 15)).values]
    df_aux.reset_index(drop=True, inplace=True)
    df_aux.plot(legend=False, ax=ax1)
    df_first_diff = df_aux - df_aux.shift()
    df_first_diff = df_first_diff[df_first_diff != 0]
    f_sharpe = df_first_diff.mean()/df_first_diff.std()
    ax1.set_title('$k = {}$ | $sharpe = {:0.2f}$'.format(s_key, f_sharpe), fontsize=10)
    ax1.xaxis.set_ticklabels([])
    ax1.set_ylabel('PnL', fontsize=8)
    ax1.set_xlabel('Time', fontsize=8)
f.tight_layout()
s_title = 'Cumulative PnL Changing K\n'
f.suptitle(s_title, fontsize=16, y=1.03);

# improving Gamma
import qtrader.eda as eda;reload(eda);
s_fname = 'log/train_test/sim_Thu_Oct__6_154516_2016.log'
get_ipython().magic("time d_rtn_gammas = eda.count_by_k_gamma(s_fname, 'LearningAgent_k', 'gamma')")

import pandas as pd
import matplotlib.pyplot as plt

f, na_ax = plt.subplots(1, 4, sharex=True, sharey=True)
for ax1, s_key in zip(na_ax.ravel(), ['0.3', '0.5', '0.7', '0.9']):
    df_aux = pd.Series(d_rtn_gammas[s_key][5])
    df_filter = pd.Series([x.hour for x in df_aux.index])
    df_aux = df_aux[((df_filter < 15)).values]
    df_aux.reset_index(drop=True, inplace=True)
    df_aux.plot(legend=False, ax=ax1)
    df_first_diff = df_aux - df_aux.shift()
    f_sharpe = df_first_diff.mean()/df_first_diff.std()
    ax1.set_title('$\gamma = {}$ | $sharpe = {:0.2f}$'.format(s_key, f_sharpe), fontsize=10)
    ax1.xaxis.set_ticklabels([])
    ax1.set_ylabel('PnL', fontsize=8)
    ax1.set_xlabel('Time Step', fontsize=8)
f.tight_layout()
s_title = 'Cumulative PnL Changing Gamma\n'
f.suptitle(s_title, fontsize=16, y=1.03);

# analyze the logs from the in-sample tests
import qtrader.eda as eda;reload(eda);
# s_fname = 'log/train_test/sim_Fri_Oct__7_002946_2016.log'  # 15 old
s_fname = 'log/train_test/sim_Wed_Oct__5_110344_2016.log'  # 15
# s_fname = 'log/train_test/sim_Thu_Oct__6_165539_2016.log'  # 25
# s_fname = 'log/train_test/sim_Thu_Oct__6_175507_2016.log'  # 35
# s_fname = 'log/train_test/sim_Thu_Oct__6_183555_2016.log'  # 5
get_ipython().magic("time d_rtn_train_2 = eda.simple_counts(s_fname, 'LearningAgent_k')")

import qtrader.eda as eda; reload(eda);
eda.plot_train_test_sim(d_rtn)

# analyze the logs from the out-of-sample tests
import qtrader.eda as eda;reload(eda);
s_fname = 'log/train_test/sim_Fri_Oct__7_003943_2016.log'  # idx = 15 old
get_ipython().magic("time d_rtn_test_1 = eda.simple_counts(s_fname, 'LearningAgent_k')")

# analyze the logs from the out-of-sample tests
import qtrader.eda as eda;reload(eda);
s_fname = 'log/train_test/sim_Wed_Oct__5_111812_2016.log'  # idx = 15
get_ipython().magic("time d_rtn_test_2 = eda.simple_counts(s_fname, 'LearningAgent_k')")

# compare the old with the data using the new configuration
import pandas as pd
df_plot = pd.DataFrame(d_rtn_test_1['pnl']['test']).mean(axis=1).fillna(method='ffill')
ax1 = df_plot.plot(legend=True, label='old')
df_plot = pd.DataFrame(d_rtn_test_2['pnl']['test']).mean(axis=1).fillna(method='ffill')
df_plot.plot(legend=True, label='new', ax=ax1)
ax1.set_title('Cumulative PnL Produced by New\nand Old Configurations')
ax1.set_xlabel('Time')
ax1.set_ylabel('PnL');

# analyze the logs from the out-of-sample tests
import qtrader.eda as eda;reload(eda);
l_fname = ['log/train_test/sim_Thu_Oct__6_171842_2016.log',  # idx = 25
           'log/train_test/sim_Thu_Oct__6_181611_2016.log',  # idx = 35
           'log/train_test/sim_Thu_Oct__6_184852_2016.log']  # idx = 5
def foo(l_fname):
    d_learning_k = {}
    for idx, s_fname in zip([25, 35, 5], l_fname):
        d_learning_k[idx] = eda.simple_counts(s_fname, 'LearningAgent_k')
    return d_learning_k

get_ipython().magic('time d_learning_k = foo(l_fname)')

import pandas as pd
import matplotlib.pyplot as plt

f, na_ax = plt.subplots(1, 3, sharey=True)
for ax1, idx in zip(na_ax.ravel(), [5, 25, 35]):
    df_plot = pd.DataFrame(d_learning_k[idx]['pnl']['test']).mean(axis=1)
    df_plot.fillna(method='ffill').plot(legend=False, ax=ax1)
    ax1.set_title('idx: {}'.format(idx + 1), fontsize=10)
    ax1.set_ylabel('PnL', fontsize=8)
    ax1.set_xlabel('Time', fontsize=8)
f.tight_layout()
s_title = 'Cumulative PnL in Diferent Days\n'
f.suptitle(s_title, fontsize=16, y=1.03);

# analyze the logs from the out-of-sample random agent
import qtrader.eda as eda;reload(eda);
s_fname = 'log/train_test/sim_Wed_Oct__5_111907_2016.log'  # idx = 15
get_ipython().magic("time d_rtn_test_1r = eda.simple_counts(s_fname, 'BasicAgent')")

import pandas as pd
import scipy
ax1 = pd.DataFrame(d_rtn_test_2['pnl']['test']).mean(axis=1).fillna(method='ffill').plot(legend=True, label='LearningAgent_k')
pd.DataFrame(d_rtn_test_1r['pnl']['test']).mean(axis=1).fillna(method='ffill').plot(legend=True, label='RandomAgent', ax=ax1)
ax1.set_title('Cumulative PnL Comparision\n')
ax1.set_xlabel('Time')
ax1.set_ylabel('PnL');
#performs t-test
a = [float(pd.DataFrame(d_rtn_test_2['pnl']['test']).iloc[-1].values)] * 2
b = list(pd.DataFrame(d_rtn_test_1r['pnl']['test']).fillna(method='ffill').iloc[-1].values)
tval, p_value = scipy.stats.ttest_ind(a, b, equal_var=False)

print "t-value = {:0.6f}, p-value = {:0.8f}".format(tval, p_value)

# analyze the logs from the out-of-sample tests
import qtrader.eda as eda;reload(eda);
l_fname = ['log/train_test/sim_Thu_Oct__6_172024_2016.log',  # idx = 25
           'log/train_test/sim_Thu_Oct__6_181735_2016.log',  # idx = 35
           'log/train_test/sim_Thu_Oct__6_184957_2016.log']  # idx = 5
def foo(l_fname):
    d_basic = {}
    for idx, s_fname in zip([25, 35, 5], l_fname):
        d_basic[idx] = eda.simple_counts(s_fname, 'BasicAgent')
    return d_basic

get_ipython().magic('time d_basic = foo(l_fname)')

import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats

f, na_ax = plt.subplots(1, 3, sharey=True)
l_stattest = []
for ax1, idx in zip(na_ax.ravel(), [5, 25, 35]):
    # plot results
    df_learning_agent = pd.DataFrame(d_learning_k[idx]['pnl']['test']).mean(axis=1)
    df_learning_agent.fillna(method='ffill').plot(legend=True, label='LearningAgent_k', ax=ax1)
    df_random_agent = pd.DataFrame(d_basic[idx]['pnl']['test']).mean(axis=1)
    df_random_agent.fillna(method='ffill').plot(legend=True, label='RandomAgent', ax=ax1)
    #performs t-test
    a = [float(pd.DataFrame(d_learning_k[idx]['pnl']['test']).iloc[-1].values)] * 2
    b = list(pd.DataFrame(d_basic[idx]['pnl']['test']).iloc[-1].values)
    tval, p_value = scipy.stats.ttest_ind(a, b, equal_var=False)
    l_stattest.append({'key': idx+1,'tval': tval, 'p_value': p_value/2})
    # set axis
    ax1.set_title('idx: ${}$ | p-value : ${:.3f}$'.format(idx+1, p_value/2.), fontsize=10)
    ax1.set_ylabel('PnL', fontsize=8)
    ax1.set_xlabel('Time', fontsize=8)
f.tight_layout()
s_title = 'Cumulative PnL Comparision in Diferent Days\n'
f.suptitle(s_title, fontsize=16, y=1.03);

pd.DataFrame(l_stattest)

# group all data generated previously
df_aux = pd.concat([pd.DataFrame(d_learning_k[5]['pnl']['test']),
                    pd.DataFrame(d_rtn_test_2['pnl']['test']),
                    pd.DataFrame(d_learning_k[25]['pnl']['test']),
                    pd.DataFrame(d_learning_k[35]['pnl']['test'])])
d_data = df_aux.to_dict()
df_plot = eda.make_df(d_data).reset_index(drop=True)[1]

df_aux = pd.concat([pd.DataFrame(d_basic[5]['pnl']['test']).mean(axis=1),
                    pd.DataFrame(d_rtn_test_1r['pnl']['test']).mean(axis=1),
                    pd.DataFrame(d_basic[25]['pnl']['test']).mean(axis=1),
                    pd.DataFrame(d_basic[35]['pnl']['test']).mean(axis=1)])
d_data = pd.DataFrame(df_aux).to_dict()
df_plot2 = eda.make_df(d_data).reset_index(drop=True)[0]
ax1 = df_plot.plot(legend=True, label='LearningAgent_k')
df_plot2.plot(legend=True, label='RandomAgent')
ax1.set_title('Cumulated PnL from Simulations\n', fontsize=16)
ax1.set_ylabel('PnL')
ax1.set_xlabel('Time Step');

((df_last_pnl)*100).sum()

#loading style sheet
from IPython.core.display import HTML
HTML( open('ipython_style.css').read())

#changing matplotlib defaults
get_ipython().magic('matplotlib inline')
import seaborn as sns
sns.set_palette("deep", desat=.6)
sns.set_context(rc={"figure.figsize": (8, 4)})
sns.set_style("whitegrid")
sns.set_palette(sns.color_palette("Set2", 10))



