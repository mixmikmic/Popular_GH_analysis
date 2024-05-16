get_ipython().magic('reload_ext autoreload')
get_ipython().magic('autoreload 2')
get_ipython().magic('matplotlib inline')
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import pandas as pd
import scipy.interpolate

from ar4_submission3 import run, get_numwells, get_wellnames

run_ml = False # run the ML estimators, if false just loads results from file
solve_rgt = False # run the RGT solver - takes about 30mins, run_ml must be True

if run_ml:
    data = run_all(solve_rgt)
else:
    data = pd.read_csv('ar4_submission3.csv')

matplotlib.style.use('ggplot')
facies_colors = ['#F4D03F', '#F5B041','#DC7633','#6E2C00', '#1B4F72','#2E86C1', '#AED6F1', '#A569BD', '#196F3D']
cmap_facies = colors.ListedColormap(facies_colors[0:len(facies_colors)], 'indexed')

def plotwellsim(data,f,y=None,title=None):
    wells = data['Well Name'].unique()
    nwells = len(wells)
    dfg = data.groupby('Well Name',sort=False)
    fig, ax = plt.subplots(nrows=1, ncols=nwells, figsize=(12, 9), sharey=True)
    if (title):
        plt.suptitle(title)
    if (y):
        miny = data[y].min()
        maxy = data[y].max()
        ax[0].set_ylabel(y)
    else:
        miny = 0
        maxy = dfg.size().max()
        ax[0].set_ylabel('Depth from top of well')
    vmin=data[f].min()
    vmax=data[f].max()
    if (f=='Facies') | (f=='NeighbFacies'):
        cmap = cmap_facies
    else:
        cmap = 'viridis'
    for wellidx,(name,group) in enumerate(dfg):
        if y:
            welldinterp = scipy.interpolate.interp1d(group[y], group[f], bounds_error=False)
            nf = len(data[y].unique())
        else:
            welldinterp = scipy.interpolate.interp1d(np.arange(0,len(group[f])), group[f], bounds_error=False)
            nf = (maxy-miny)/0.5
        fnew = np.linspace(miny, maxy, nf) 
        ynew = welldinterp(fnew)
        ynew = ynew[:, np.newaxis]
        ax[wellidx].set_xticks([])
        ax[wellidx].set_yticks([])
        ax[wellidx].grid(False)
        ax[wellidx].imshow(ynew, aspect='auto',vmin=vmin,vmax=vmax,cmap=cmap)
        ax[wellidx].set_xlabel(name,rotation='vertical')

well_width = 100
mind = data['Depth'].min()
maxd = data['Depth'].max()
fim = np.nan*np.ones([int((maxd-mind)*2)+1,get_numwells(data)*well_width])
dfg = data.groupby('Well Name',sort=False)
plt.figure(figsize=(12, 9))
plt.title('Wells cover different depths')
ax=plt.subplot(111)
ax.grid(False)
ax.get_xaxis().set_visible(False)
ax.set_ylabel('Depth')
plt.tick_params(axis="both", which="both", bottom="off", top="off", 
                labelbottom="off", left="off", right="off", labelleft="off")
ax.text(well_width*7,1000,'Colours represent\nformation class')
for i,(name,group) in enumerate(dfg):
    if (maxd-group['Depth'].max())*2 > 600:
        ty = (maxd-group['Depth'].max())*2-50
        tva='bottom'
    else:
        ty = (maxd-group['Depth'].min())*2+50
        tva='top'
    ax.text(well_width*(i+0.5),ty,name,va=tva,rotation='vertical')
    for j in range(len(group)):
        fim[-int((group.loc[group.index[j],'Depth']-maxd)*2),i*well_width:(i+1)*well_width]=group.loc[group.index[j],'FormationClass']
    
plt.imshow(fim,cmap='viridis',aspect='auto')

ax=data.groupby('Well Name',sort=False)['Formation3Depth'].first().plot.bar(title='Top of Formation 3 Depth')
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()
ax.grid(axis='x')

plotwellsim(data,'RELPOS',title='RELPOS by depth')
plotwellsim(data,'RELPOS','RGT',title='RELPOS by RGT')
plotwellsim(data,'GR',title='GR by depth')
plotwellsim(data,'GR','RGT',title='GR by RGT')
plotwellsim(data,'Facies',title='Facies by depth')
plotwellsim(data,'Facies','RGT',title='Facies by RGT')

ax=plt.subplot(111)
plt.axis('off')
plt.ylim(-1,1)
plt.title('Well Position (X1D)')
for i,wellname in enumerate(get_wellnames(data)):
    x=data.loc[data['Well Name']==wellname, 'X1D'].values[0]
    plt.scatter(x=-x,y=0)
    if i%2==0:
        tva='top'
        ty=-0.1
    else:
        tva='bottom'
        ty=0.1
    plt.text(-x,ty,wellname,rotation='vertical',ha='center',va=tva)

plotwellsim(data,'NeighbFacies','RGT','Facies found by KNN classifier')
plotwellsim(data,'Facies','RGT','True facies (except validation wells)')

plt.figure(figsize=(12, 9))
plt.tick_params(axis="x", which="both", bottom="off", top="off", 
                labelbottom="off", left="off", right="off", labelleft="off")
wellnames = get_wellnames(data).tolist()
num_wells = get_numwells(data)
formations = data['FormationClass'].unique().tolist()
num_formations = len(formations)
colors=plt.cm.viridis(np.linspace(0,1,len(wellnames)))
dfg=data.groupby(['Well Name','FormationClass'], sort=False)
for i,(name,group) in enumerate(dfg):
    widx = wellnames.index(name[0])
    fidx = formations.index(name[1])
    plt.bar(fidx*num_wells+widx,group['FormationClassCompaction'].values[0],color=colors[widx])
plt.ylim(0,0.1)
plt.title('Formation compaction')
plt.text(70,0.09,'Color represents well, each cycle along the x axis is one formation')

v_rows = (data['Well Name'] == 'STUART') | (data['Well Name'] == 'CRAWFORD')
plotwellsim(data.loc[v_rows,:],'Facies',title='Predicted facies')

