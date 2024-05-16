get_ipython().magic('pylab inline')
import scipy.io.netcdf

prog_file = scipy.io.netcdf_file('prog__0001_006.nc')
prog_file.variables

e_handle = prog_file.variables['e']
print('Description =', e_handle.long_name)
print('Shape =',e_handle.shape)

plt.pcolormesh( e_handle[0,0] )

plt.pcolormesh( e_handle[0,0], cmap=cm.seismic ); plt.colorbar();

import ipywidgets

[e_handle[:,0].min(), e_handle[:,0].max()]

def plot_ssh(record):
    plt.pcolormesh( e_handle[record,0], cmap=cm.spectral )
    plt.clim(-.5,.8) # Fixed scale here
    plt.colorbar()

ipywidgets.interact(plot_ssh, record=(0,e_handle.shape[0]-1,1));

from IPython import display

for n in range( e_handle.shape[0]):
    display.display(plt.gcf())
    plt.clf()
    plot_ssh(n)
    display.clear_output(wait=True)

