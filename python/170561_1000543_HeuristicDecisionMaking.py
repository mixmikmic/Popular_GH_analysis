import pymc3 as pm
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
get_ipython().run_line_magic('qtconsole', '--colors=linux')
plt.style.use('ggplot')

from matplotlib import gridspec
from theano import tensor as tt
from scipy import stats

import scipy.io as sio
matdata = sio.loadmat('data/StopSearchData.mat')

y = np.squeeze(matdata['y'])
m = np.squeeze(np.float32(matdata['m']))
p = np.squeeze(matdata['p'])
v = np.squeeze(np.float32(matdata['v']))
x = np.squeeze(np.float32(matdata['x']))

# Constants
n, nc = np.shape(m)  # number of stimuli and cues
nq, _ = np.shape(p)  # number of questions
ns, _ = np.shape(y)  # number of subjects

s = np.argsort(v)  # s[1:nc] <- rank(v[1:nc])
t = []
# TTB Model For Each Question
for q in range(nq):
    # Add Cue Contributions To Mimic TTB Decision
    tmp1 = np.zeros(nc)
    for j in range(nc):
        tmp1[j] = (m[p[q, 0]-1, j]-m[p[q, 1]-1, j])*np.power(2, s[j])
    # Find if Cue Favors First, Second, or Neither Stimulus
    tmp2 = np.sum(tmp1)
    tmp3 = -1*np.float32(-tmp2 > 0)+np.float32(tmp2 > 0)
    t.append(tmp3+1)
t = np.asarray(t, dtype=int)
tmat = np.tile(t[np.newaxis, :], (ns, 1))

with pm.Model() as model1:
    gamma = pm.Uniform('gamma', lower=.5, upper=1)
    gammat = tt.stack([1-gamma, .5, gamma])
    yiq = pm.Bernoulli('yiq', p=gammat[tmat], observed=y)
    trace1 = pm.sample(3e3, njobs=2, tune=1000)
    
pm.traceplot(trace1);

ppc = pm.sample_ppc(trace1, samples=100, model=model1)
yiqpred = np.asarray(ppc['yiq'])
fig = plt.figure(figsize=(16, 8))
x1 = np.repeat(np.arange(ns)+1, nq).reshape(ns, -1).flatten()
y1 = np.repeat(np.arange(nq)+1, ns).reshape(nq, -1).T.flatten()

plt.scatter(y1, x1, s=np.mean(yiqpred, axis=0)*200, c='w')
plt.scatter(y1[y.flatten() == 1], x1[y.flatten() == 1], marker='x', c='r')
plt.plot(np.ones(100)*24.5, np.linspace(0, 21, 100), '--', lw=1.5, c='k')
plt.axis([0, 31, 0, 21])
plt.show()

# Question cue contributions template
qcc = np.zeros((nq, nc))
for q in range(nq):
    # Add Cue Contributions To Mimic TTB Decision
    for j in range(nc):
        qcc[q, j] = (m[p[q, 0]-1, j]-m[p[q, 1]-1, j])

qccmat = np.tile(qcc[np.newaxis, :, :], (ns, 1, 1))
# TTB Model For Each Question
s = np.argsort(v)  # s[1:nc] <- rank(v[1:nc])
smat = np.tile(s[np.newaxis, :], (ns, nq, 1))
ttmp = np.sum(qccmat*np.power(2, smat), axis=2)
tmat = -1*(-ttmp > 0)+(ttmp > 0)+1
t = tmat[0]
# tmat = np.tile(t[np.newaxis, :], (ns, 1))        

# WADD Model For Each Question
xmat = np.tile(x[np.newaxis, :], (ns, nq, 1))
wtmp = np.sum(qccmat*xmat, axis=2)
wmat = -1*(-wtmp > 0)+(wtmp > 0)+1
w = wmat[0]

print(t)
print(w)

with pm.Model() as model2:
    phi = pm.Beta('phi', alpha=1, beta=1, testval=.01)
    
    zi = pm.Bernoulli('zi', p=phi, shape=ns,
                      testval=np.asarray([1,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0]))
    zi_ = tt.reshape(tt.repeat(zi, nq), (ns, nq))
    
    gamma = pm.Uniform('gamma', lower=.5, upper=1)
    gammat = tt.stack([1-gamma, .5, gamma])

    t2 = tt.switch(tt.eq(zi_, 1), tmat, wmat)
    yiq = pm.Bernoulli('yiq', p=gammat[t2], observed=y)

    trace2 = pm.sample(3e3, njobs=2, tune=1000)
    
pm.traceplot(trace2);

fig = plt.figure(figsize=(16, 4))
zitrc = trace2['zi'][1000:]
plt.bar(np.arange(ns)+1, 1-np.mean(zitrc, axis=0))
plt.yticks([0, 1], ('TTB', 'WADD'))
plt.xlabel('Subject')
plt.ylabel('Group')
plt.axis([0, 21, 0, 1])
plt.show()

with pm.Model() as model3:
    gamma = pm.Uniform('gamma', lower=.5, upper=1)
    gammat = tt.stack([1-gamma, .5, gamma])
    
    v1 = pm.HalfNormal('v1', sd=1, shape=ns*nc)    
    s1 = pm.Deterministic('s1', tt.argsort(v1.reshape((ns, 1, nc)), axis=2))
    smat2 = tt.tile(s1, (1, nq, 1))  # s[1:nc] <- rank(v[1:nc])
    
    # TTB Model For Each Question
    ttmp = tt.sum(qccmat*tt.power(2, smat2), axis=2)
    tmat = -1*(-ttmp > 0)+(ttmp > 0)+1
    
    yiq = pm.Bernoulli('yiq', p=gammat[tmat], observed=y)

with model3:
    # trace3 = pm.sample(3e3, njobs=2, tune=1000)
    trace3 = pm.sample(1e5, step=pm.Metropolis(), njobs=2)

pm.traceplot(trace3, varnames=['gamma', 'v1']);

burnin = 50000
# v1trace = np.squeeze(trace3['v1'][burnin:])
# s1trace = np.argsort(v1trace, axis=2)
s1trace = np.squeeze(trace3[burnin:]['s1'])
for subj_id in [12, 13]:
    subj_s = np.squeeze(s1trace[:,subj_id-1,:])
    unique_ord = np.vstack({tuple(row) for row in subj_s})
    num_display = 10
    print('Subject %s' %(subj_id))
    print('There are %s search orders sampled in the posterior.'%(unique_ord.shape[0]))
    mass_ = []
    for s_ in unique_ord:
        mass_.append(np.mean(np.sum(subj_s == s_, axis=1) == len(s_)))
    mass_ = np.asarray(mass_)
    sortmass = np.argsort(mass_)[::-1]
    for i in sortmass[:num_display]:
        s_ = unique_ord[i]
        print('Order=(' + str(s_+1) + '), Estimated Mass=' + str(mass_[i]))

with pm.Model() as model4:
    phi = pm.Beta('phi', alpha=1, beta=1, testval=.01)
    
    zi = pm.Bernoulli('zi', p=phi, shape=ns,
                      testval=np.asarray([1,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0]))
    zi_ = tt.reshape(tt.repeat(zi, nq), (ns, nq))
    
    gamma = pm.Uniform('gamma', lower=.5, upper=1)
    gammat = tt.stack([1-gamma, .5, gamma])

    v1 = pm.HalfNormal('v1', sd=1, shape=ns*nc)    
    s1 = pm.Deterministic('s1', tt.argsort(v1.reshape((ns, 1, nc)), axis=2))
    smat2 = tt.tile(s1, (1, nq, 1))  # s[1:nc] <- rank(v[1:nc])
    
    # TTB Model For Each Question
    ttmp = tt.sum(qccmat*tt.power(2, smat2), axis=2)
    tmat = -1*(-ttmp > 0) + (ttmp > 0) + 1
            
    t2 = tt.switch(tt.eq(zi_, 1), tmat, wmat)
    yiq = pm.Bernoulli('yiq', p=gammat[t2], observed=y)
    
    trace4 = pm.sample(1e5, step=pm.Metropolis())
    
burnin=50000
pm.traceplot(trace4[burnin:], varnames=['phi', 'gamma']);

ppc = pm.sample_ppc(trace4[burnin:], samples=100, model=model4)
yiqpred = np.asarray(ppc['yiq'])
fig = plt.figure(figsize=(16, 8))
x1 = np.repeat(np.arange(ns)+1, nq).reshape(ns, -1).flatten()
y1 = np.repeat(np.arange(nq)+1, ns).reshape(nq, -1).T.flatten()

plt.scatter(y1, x1, s=np.mean(yiqpred, axis=0)*200, c='w')
plt.scatter(y1[y.flatten() == 1], x1[y.flatten() == 1], marker='x', c='r')
plt.plot(np.ones(100)*24.5, np.linspace(0, 21, 100), '--', lw=1.5, c='k')
plt.axis([0, 31, 0, 21])
plt.show()

