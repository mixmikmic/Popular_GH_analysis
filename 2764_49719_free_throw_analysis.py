from __future__ import division

import pandas as pd
import numpy as np
import scipy as sp

import scipy.stats

import toyplot as tp

df = pd.read_csv('free_throws.csv', names=["away", "home", "team", "player", "score"])

df["at_home"] = df["home"] == df["team"]

df.head()

df.groupby("at_home").mean()

sdf = pd.pivot_table(df, index=["player", "team"], columns="at_home", values=["score"], 
               aggfunc=[len, sum], fill_value=0).reset_index()
sdf.columns = ['player', 'team', 'atm_away', 'atm_home', 'score_away', 'score_home']

sdf['atm_total'] = sdf['atm_away'] + sdf['atm_home']
sdf['score_total'] = sdf['score_away'] + sdf['score_home']

sdf.sample(10)

data = sdf.query('atm_total > 50').copy()
len(data)

data['p_home'] = data['score_home'] / data['atm_home']
data['p_away'] = data['score_away'] / data['atm_away']
data['p_ovr'] = (data['score_total']) / (data['atm_total'])

# two-sided
data['zval'] = (data['p_home'] - data['p_away']) / np.sqrt(data['p_ovr'] * (1-data['p_ovr']) * (1/data['atm_away'] + 1/data['atm_home']))
data['pval'] = 2*(1-sp.stats.norm.cdf(np.abs(data['zval'])))

# one-sided testing home advantage
# data['zval'] = (data['p_home'] - data['p_away']) / np.sqrt(data['p_ovr'] * (1-data['p_ovr']) * (1/data['atm_away'] + 1/data['atm_home']))
# data['pval'] = (1-sp.stats.norm.cdf(data['zval']))

data.sample(10)

canvas = tp.Canvas(800, 300)
ax1 = canvas.axes(grid=(1, 2, 0), label="Histogram p-values")
hist_p = ax1.bars(np.histogram(data["pval"], bins=50, normed=True), color="steelblue")
hisp_p_density = ax1.plot([0, 1], [1, 1], color="red")
ax2 = canvas.axes(grid=(1, 2, 1), label="Histogram z-values")
hist_z = ax2.bars(np.histogram(data["zval"], bins=50, normed=True), color="orange")
x = np.linspace(-3, 3, 200)
hisp_z_density = ax2.plot(x, sp.stats.norm.pdf(x), color="red")

T = -2 * np.sum(np.log(data["pval"]))
print 'p-value for Fisher Combination Test: {:.3e}'.format(1 - sp.stats.chi2.cdf(T, 2*len(data)))

print '"p-value" Bonferroni: {:.3e}'.format(min(1, data["pval"].min() * len(data)))

alpha = 0.05
data["reject_naive"] = 1*(data["pval"] < alpha)

print 'Number of rejections: {}'.format(data["reject_naive"].sum())

from statsmodels.sandbox.stats.multicomp import multipletests

data["reject_bc"] = 1*(data["pval"] < alpha / len(data))
print 'Number of rejections: {}'.format(data["reject_bc"].sum())

is_reject, corrected_pvals, _, _ = multipletests(data["pval"], alpha=0.1, method='fdr_bh')

data["reject_fdr"] = 1*is_reject
data["pval_fdr"] = corrected_pvals
print 'Number of rejections: {}'.format(data["reject_fdr"].sum())

len(data.query("atm_total > 2500"))

reduced_data = data.query("atm_total > 2500").copy()

is_reject2, corrected_pvals2, _, _ = multipletests(reduced_data["pval"], alpha=0.1, method='fdr_bh')
reduced_data["reject_fdr2"] = 1*is_reject2
reduced_data["pval_fdr2"] = corrected_pvals2


print 'Number of rejections: {}'.format(reduced_data["reject_fdr2"].sum())

