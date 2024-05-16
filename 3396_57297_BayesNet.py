import os,sys
import numpy
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
sys.path.insert(0,'../')
from utils.mkdesign import create_design_singlecondition
from nipy.modalities.fmri.hemodynamic_models import spm_hrf,compute_regressor
from utils.make_data import make_continuous_data
from utils.graph_utils import show_graph_from_adjmtx,show_graph_from_pattern
from statsmodels.tsa.arima_process import arma_generate_sample
import scipy.stats
from dcm_sim import sim_dcm_dataset

results_dir = os.path.abspath("../results")
if not os.path.exists(results_dir):
    os.mkdir(results_dir)

_,data_conv,params=sim_dcm_dataset(verbose=True)

A_mtx=params['A']
B_mtx=params['B']
u=params['u']

# downsample design to 1 second TR
u=numpy.convolve(params['u'],spm_hrf(params['stepsize'],oversampling=1))
u=u[range(0,data_conv.shape[0],int(1./params['stepsize']))]
ntp=u.shape[0]



tetrad_dir='/home/vagrant/data/tetrad_files'
if not os.path.exists(tetrad_dir):
    os.mkdir(tetrad_dir)

nfiles=10
for i in range(nfiles):
    _,data_conv,params=sim_dcm_dataset()


    # downsample to 1 second TR
    data=data_conv[range(0,data_conv.shape[0],int(1./params['stepsize']))]
    ntp=data.shape[0]

    imagesdata=numpy.hstack((numpy.array(u)[:,numpy.newaxis],data))
    numpy.savetxt(os.path.join(tetrad_dir,"data%03d.txt"%i),
              imagesdata,delimiter='\t',
             header='u\t0\t1\t2\t3\t4',comments='')

get_ipython().system('bash run_images.sh')

g=show_graph_from_pattern('images_test/test.pattern.dot')

show_graph_from_adjmtx(A_mtx,B_mtx,params['C'])



