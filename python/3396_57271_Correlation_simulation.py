import numpy
import nibabel
import os
import nilearn.plotting
import matplotlib.pyplot as plt
from statsmodels.regression.linear_model import OLS
import nipype.interfaces.fsl as fsl
import scipy.stats

if not 'FSLDIR' in os.environ.keys():
    raise Exception('This notebook requires that FSL is installed and the FSLDIR environment variable is set')

get_ipython().magic('matplotlib inline')

pthresh=0.001  # cluster forming threshold
cthresh=10     # cluster extent threshold
nsubs=28       # number of subjects

recreate_paper_figure=False
if recreate_paper_figure:
    seed=6636
else:
    seed=numpy.ceil(numpy.random.rand()*100000).astype('int')
    print(seed)

numpy.random.seed(seed)

maskimg=os.path.join(os.getenv('FSLDIR'),'data/standard/MNI152_T1_2mm_brain_mask.nii.gz')
mask=nibabel.load(maskimg)
maskdata=mask.get_data()
maskvox=numpy.where(maskdata>0)
print('Mask includes %d voxels'%len(maskvox[0]))

imgmean=1000    # mean activation within mask
imgstd=100      # standard deviation of noise within mask
behavmean=100   # mean of behavioral regressor
behavstd=1      # standard deviation of behavioral regressor

data=numpy.zeros((maskdata.shape + (nsubs,)))

for i in range(nsubs):
    tmp=numpy.zeros(maskdata.shape)
    tmp[maskvox]=numpy.random.randn(len(maskvox[0]))*imgstd+imgmean
    data[:,:,:,i]=tmp

newimg=nibabel.Nifti1Image(data,mask.get_affine(),mask.get_header())
newimg.to_filename('fakedata.nii.gz')
regressor=numpy.random.randn(nsubs,1)*behavstd+behavmean
numpy.savetxt('regressor.txt',regressor)

smoothing_fwhm=6 # FWHM in millimeters

smooth=fsl.IsotropicSmooth(fwhm=smoothing_fwhm,
                           in_file='fakedata.nii.gz',
                           out_file='fakedata_smooth.nii.gz')
smooth.run()

glm = fsl.GLM(in_file='fakedata_smooth.nii.gz', 
              design='regressor.txt', 
              out_t_name='regressor_tstat.nii.gz',
             demean=True)
glm.run()

tcut=scipy.stats.t.ppf(1-pthresh,nsubs-1)
cl = fsl.Cluster()
cl.inputs.threshold = tcut
cl.inputs.in_file = 'regressor_tstat.nii.gz'
cl.inputs.out_index_file='tstat_cluster_index.nii.gz'
results=cl.run()

clusterimg=nibabel.load(cl.inputs.out_index_file)
clusterdata=clusterimg.get_data()
indices=numpy.unique(clusterdata)

clustersize=numpy.zeros(len(indices))
clustermean=numpy.zeros((len(indices),nsubs))
indvox={}
for c in range(1,len(indices)):
    indvox[c]=numpy.where(clusterdata==c)    
    clustersize[c]=len(indvox[c][0])
    for i in range(nsubs):
        tmp=data[:,:,:,i]
        clustermean[c,i]=numpy.mean(tmp[indvox[c]])
corr=numpy.corrcoef(regressor.T,clustermean[-1])

print('Found %d clusters exceeding p<%0.3f and %d voxel extent threshold'%(c,pthresh,cthresh))
print('Largest cluster: correlation=%0.3f, extent = %d voxels'%(corr[0,1],len(indvox[c][0])))

# set cluster to show - 0 is the largest, 1 the second largest, and so on
cluster_to_show=0

# translate this variable into the index of indvox
cluster_to_show_idx=len(indices)-cluster_to_show-1

# plot the (circular) relation between fMRI signal and 
# behavioral regressor in the chosen cluster

plt.scatter(regressor.T,clustermean[cluster_to_show_idx])
plt.title('Correlation = %0.3f'%corr[0,1],fontsize=14)
plt.xlabel('Fake behavioral regressor',fontsize=18)
plt.ylabel('Fake fMRI data',fontsize=18)
m, b = numpy.polyfit(regressor[:,0], clustermean[cluster_to_show_idx], 1)
axes = plt.gca()
X_plot = numpy.linspace(axes.get_xlim()[0],axes.get_xlim()[1],100)
plt.plot(X_plot, m*X_plot + b, '-')
plt.savefig('scatter.png',dpi=600)

tstat=nibabel.load('regressor_tstat.nii.gz').get_data()
thresh_t=clusterdata.copy()
cutoff=numpy.min(numpy.where(clustersize>cthresh))
thresh_t[thresh_t<cutoff]=0
thresh_t=thresh_t*tstat
thresh_t_img=nibabel.Nifti1Image(thresh_t,mask.get_affine(),mask.get_header())

mid=len(indvox[cluster_to_show_idx][0])/2
coords=numpy.array([indvox[cluster_to_show_idx][0][mid],
                    indvox[cluster_to_show_idx][1][mid],
                    indvox[cluster_to_show_idx][2][mid],1]).T
mni=mask.get_qform().dot(coords)
nilearn.plotting.plot_stat_map(thresh_t_img,
        os.path.join(os.getenv('FSLDIR'),'data/standard/MNI152_T1_2mm_brain.nii.gz'),
                              threshold=cl.inputs.threshold,
                               cut_coords=mni[:3])
plt.savefig('slices.png',dpi=600)

