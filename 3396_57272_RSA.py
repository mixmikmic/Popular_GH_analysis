import numpy
import nibabel
import os
from haxby_data import HaxbyData
from nilearn.input_data import NiftiMasker
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import sklearn.manifold
import scipy.cluster.hierarchy

datadir='/home/vagrant/nilearn_data/haxby2001/subj2'

print('Using data from %s'%datadir)

haxbydata=HaxbyData(datadir)

modeldir=os.path.join(datadir,'blockmodel')
try:
    os.chdir(modeldir)
except:
    print('problem changing to %s'%modeldir)
    print('you may need to run the Classification Analysis script first')
    

use_whole_brain=False

if use_whole_brain:
    maskimg=haxbydata.brainmaskfile
else:
    maskimg=haxbydata.vtmaskfile
    
nifti_masker = NiftiMasker(mask_img=maskimg, standardize=False)
fmri_masked = nifti_masker.fit_transform(os.path.join(modeldir,'zstatdata.nii.gz'))

print(ci,cj,i,j)

cc=numpy.zeros((8,8,12,12))

# loop through conditions
for ci in range(8):
    for cj in range(8):
        for i in range(12):
            for j in range(12):
                if i==6 or j==6:  # problem with run 6 - skip it
                    continue
                idx=numpy.where(numpy.logical_and(haxbydata.runs==i,haxbydata.condnums==ci+1))
                if len(idx[0])>0:
                    idx_i=idx[0][0]
                else:
                    print('problem',ci,cj,i,j)
                    idx_i=None
                idx=numpy.where(numpy.logical_and(haxbydata.runs==j,haxbydata.condnums==cj+1))
                if len(idx[0])>0:
                    idx_j=idx[0][0]
                else:
                    print('problem',ci,cj,i,j)
                    idx_j=None
                if not idx_i is None and not idx_j is None:
                    cc[ci,cj,i,j]=numpy.corrcoef(fmri_masked[idx_i,:],fmri_masked[idx_j,:])[0,1]
                else:
                    cc[ci,cj,i,j]=numpy.nan
meansim=numpy.zeros((8,8))
for ci in range(8):
    for cj in range(8):
        cci=cc[ci,cj,:,:]
        meansim[ci,cj]=numpy.nanmean(numpy.hstack((cci[numpy.triu_indices(12,1)],
                                            cci[numpy.tril_indices(12,1)])))

plt.imshow(meansim,interpolation='nearest')
plt.colorbar()

l=scipy.cluster.hierarchy.ward(1.0 - meansim)

cl=scipy.cluster.hierarchy.dendrogram(l,labels=haxbydata.condlabels,orientation='right')

# within-condition

face_corr={}
corr_means=[]
corr_stderr=[]
corr_stimtype=[]
for k in haxbydata.cond_dict.keys():
    face_corr[k]=[]
    for i in range(12):
        for j in range(12):
            if i==6 or j==6:
                continue
            if i==j:
                continue
            face_corr[k].append(cc[haxbydata.cond_dict['face']-1,haxbydata.cond_dict[k]-1,i,j])

    corr_means.append(numpy.mean(face_corr[k]))
    corr_stderr.append(numpy.std(face_corr[k])/numpy.sqrt(len(face_corr[k])))
    corr_stimtype.append(k)

idx=numpy.argsort(corr_means)[::-1]
plt.bar(numpy.arange(0.5,8.),[corr_means[i] for i in idx],yerr=[corr_stderr[i] for i in idx]) #,yerr=corr_sterr[idx])
t=plt.xticks(numpy.arange(1,9), [corr_stimtype[i] for i in idx],rotation=70)
plt.ylabel('Mean between-run correlation with faces')

import sklearn.manifold
mds=sklearn.manifold.MDS()
#mds=sklearn.manifold.TSNE(early_exaggeration=10,perplexity=70,learning_rate=100,n_iter=5000)
encoding=mds.fit_transform(fmri_masked)

plt.figure(figsize=(12,12))
ax=plt.axes() #[numpy.min(encoding[0]),numpy.max(encoding[0]),numpy.min(encoding[1]),numpy.max(encoding[1])])
ax.scatter(encoding[:,0],encoding[:,1])
offset=0.01
for i in range(encoding.shape[0]):
    ax.annotate(haxbydata.conditions[i].split('-')[0],(encoding[i,0],encoding[i,1]),xytext=[encoding[i,0]+offset,encoding[i,1]+offset])
#for i in range(encoding.shape[0]):
#    plt.text(encoding[i,0],encoding[i,1],'%d'%haxbydata.condnums[i])

mdsmeans=numpy.zeros((2,8))
for i in range(8):
    mdsmeans[:,i]=numpy.mean(encoding[haxbydata.condnums==(i+1),:],0)

for i in range(2):
    print('Dimension %d:'%int(i+1))
    idx=numpy.argsort(mdsmeans[i,:])
    for j in idx:
        print('%s:\t%f'%(haxbydata.condlabels[j],mdsmeans[i,j]))
    print('')






