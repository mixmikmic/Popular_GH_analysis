import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.tsa.setar_model as setar_model
import statsmodels.tsa.bds as bds

print sm.datasets.sunspots.NOTE

dta = sm.datasets.sunspots.load_pandas().data

dta.index = pd.Index(sm.tsa.datetools.dates_from_range('1700', '2008'))
del dta["YEAR"]

dta.plot(figsize=(12,8));

arma_mod30 = sm.tsa.ARMA(dta, (3,0)).fit()
print arma_mod30.params
print arma_mod30.aic, arma_mod30.bic, arma_mod30.hqic

bds.bds(arma_mod30.resid, 3)

setar_mod23 = setar_model.SETAR(dta, 2, 3).fit()
print setar_mod23.summary()

setar_mod23 = setar_model.SETAR(dta, 2, 3).fit()
f_stat, pvalue, bs_f_stats = setar_mod23.order_test() # by default tests against SETAR(1)
print pvalue

f_stat_h, pvalue_h, bs_f_stats_h = setar_mod23.order_test(heteroskedasticity='g')
print pvalue

print bds.bds(setar_mod23.resid, 3)
setar_mod23.resid.plot(figsize=(10,5));

setar_mod23.plot_threshold_ci(0, figwidth=10, figheight=5);

predict_arma_mod30 = arma_mod30.predict('1990', '2012', dynamic=True)
predict_setar_mod23 = setar_mod23.predict('1990', '2012', dynamic=True)

ax = dta.ix['1950':].plot(figsize=(12,8))
ax = predict_arma_mod30.plot(ax=ax, style='r--', linewidth=2, label='AR(3) Dynamic Prediction');
ax = predict_setar_mod23.plot(ax=ax, style='k--', linewidth=2, label='SETAR(2;3,3) Dynamic Prediction');
ax.legend();
ax.axis((-20.0, 38.0, -4.0, 200.0));

def rmsfe(y, yhat):
    return (y.sub(yhat)**2).mean()
print 'AR(3):        ', rmsfe(dta.SUNACTIVITY, predict_arma_mod30)
print 'SETAR(2;3,3): ', rmsfe(dta.SUNACTIVITY, predict_setar_mod23)

predict_arma_mod30_long = arma_mod30.predict('1960', '2012', dynamic=True)
predict_setar_mod23_long = setar_mod23.predict('1960', '2012', dynamic=True)
ax = dta.ix['1950':].plot(figsize=(12,8))
ax = predict_arma_mod30_long.plot(ax=ax, style='r--', linewidth=2, label='AR(3) Dynamic Prediction');
ax = predict_setar_mod23_long.plot(ax=ax, style='k--', linewidth=2, label='SETAR(2;3,3) Dynamic Prediction');
ax.legend();
ax.axis((-20.0, 38.0, -4.0, 200.0));

print 'AR(3):        ', rmsfe(dta.SUNACTIVITY, predict_arma_mod30_long)
print 'SETAR(2;3,3): ', rmsfe(dta.SUNACTIVITY, predict_setar_mod23_long)

