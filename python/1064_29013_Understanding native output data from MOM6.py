get_ipython().magic('pylab inline')
import scipy.io.netcdf

layer_file = scipy.io.netcdf_file('layer/prog.nc')
rho_file = scipy.io.netcdf_file('rho/prog.nc')
sigma_file = scipy.io.netcdf_file('sigma/prog.nc')
z_file = scipy.io.netcdf_file('z/prog.nc')
for v in layer_file.variables:
    print(v,layer_file.variables[v].shape,layer_file.variables[v].long_name)

get_ipython().system('head -15 layer/diag_table')

# Use the CF dimension-variable as the horizontal coordinate
xh = layer_file.variables['xh'][:] # This is the coordinate of the cell centers (h-points in 1D)
xq = layer_file.variables['xq'][:] # This is the coordinate of the cell corners (u-points in 1D)
xq = numpy.concatenate(([2*xq[0]-xq[1]],xq)) # Inserts left most edge of domain in to u-point coordinates

geom_file = scipy.io.netcdf_file('layer/ocean_geometry.nc')
for v in geom_file.variables:
    print(v,geom_file.variables[v].shape,geom_file.variables[v].long_name)

plt.plot( xh, '.', label='xh (CF 1D)');
plt.plot( geom_file.variables['lonh'][:].T, '.', label='lonh (1D)');
plt.plot( geom_file.variables['geolon'][:].T, '.', label='geolon (2D)');
plt.legend(loc='lower right');

plt.plot( geom_file.variables['D'][0,:]); plt.title('Depth');

print("mean square |h(j=0)-h(j=3)|^2 =",
      (( layer_file.variables['h'][-1,:,0,:]-layer_file.variables['h'][-1,:,3,:] )**2).sum() )

print( layer_file.variables['zl'].long_name, layer_file.variables['zl'].units, layer_file.variables['zl'][:] )

print( z_file.variables['zl'].long_name, z_file.variables['zl'].units, z_file.variables['zl'][:] )

plt.figure(figsize=(12,6))
plt.subplot(221);
plt.pcolormesh( layer_file.variables['salt'][0,:,0,:] ); plt.colorbar(); plt.title('a) Layer mode S');
plt.subplot(222);
plt.pcolormesh( rho_file.variables['salt'][0,:,0,:] ); plt.colorbar(); plt.title(r'b) $\rho$-coordinate S');
plt.subplot(223);
plt.pcolormesh( sigma_file.variables['salt'][0,:,0,:] ); plt.colorbar(); plt.title(r'c) $\sigma$-coordinate S');
plt.subplot(224);
plt.pcolormesh( z_file.variables['salt'][0,:,0,:] ); plt.colorbar(); plt.title('d) z*-coordinate S');

plt.figure(figsize=(12,6))
plt.subplot(221);
plt.pcolormesh( layer_file.variables['h'][0,:,0,:] ); plt.colorbar(); plt.title('a) Layer mode h');
plt.subplot(222);
plt.pcolormesh( rho_file.variables['h'][0,:,0,:] ); plt.colorbar(); plt.title(r'b) $\rho$-coordinate h');
plt.subplot(223);
plt.pcolormesh( sigma_file.variables['h'][0,:,0,:] ); plt.colorbar(); plt.title(r'c) $\sigma$-coordinate h');
plt.subplot(224);
plt.pcolormesh( z_file.variables['h'][0,:,0,:] ); plt.colorbar(); plt.title('d) z*-coordinate h');

plt.plot( layer_file.variables['h'][0,:,0,:].sum(axis=0), label='Layer');
plt.plot( rho_file.variables['h'][0,:,0,:].sum(axis=0), label=r'$\rho$');
plt.plot( sigma_file.variables['h'][0,:,0,:].sum(axis=0), label=r'$\sigma$');
plt.plot( z_file.variables['h'][0,:,0,:].sum(axis=0), label='z*');
plt.legend(loc='lower right'); plt.title('Coloumn total thickness');

plt.figure(figsize=(12,6))
plt.subplot(221); plt.plot( layer_file.variables['e'][0,:,0,:].T); plt.title('a) Layer mode e');
plt.subplot(222); plt.plot( rho_file.variables['e'][0,:,0,:].T); plt.title(r'b) $\rho$-coordinate e');
plt.subplot(223); plt.plot( sigma_file.variables['e'][0,:,0,:].T); plt.title(r'c) $\sigma$-coordinate e');
plt.subplot(224); plt.plot( z_file.variables['e'][0,:,0,:].T); plt.title('d) z*-coordinate e');

plt.figure(figsize=(12,6))
xl=5,12; yl=-1000,10
plt.subplot(221); plt.plot( layer_file.variables['e'][0,:,0,:].T); plt.xlim(xl); plt.ylim(yl); plt.title('a) Layer mode e');
plt.subplot(222); plt.plot( rho_file.variables['e'][0,:,0,:].T); plt.xlim(xl); plt.ylim(yl); plt.title(r'b) $\rho$-coordinate e');
plt.subplot(223); plt.plot( sigma_file.variables['e'][0,:,0,:].T); plt.xlim(xl); plt.ylim(yl); plt.title(r'c) $\sigma$-coordinate e');
plt.subplot(224); plt.plot( z_file.variables['e'][0,:,0,:].T); plt.xlim(xl); plt.ylim(yl); plt.title('d) z*-coordinate e');

plt.figure(figsize=(12,6))
xxl=50,120 # This is the zoomed-in region around the shelf break in model coordinates
plt.subplot(221)
z = ( layer_file.variables['e'][0,:-1,0,:] + layer_file.variables['e'][0,1:,0,:] ) / 2
x = xh + 0*z
plt.contourf( x, z, layer_file.variables['salt'][0,:,0,:]); plt.xlim(xxl); plt.ylim(yl); plt.title('a) Layer mode S');
plt.plot( xh, layer_file.variables['e'][0,:,0,:].T, 'k');
plt.subplot(222)
z = ( rho_file.variables['e'][0,:-1,0,:] + rho_file.variables['e'][0,1:,0,:] ) / 2
plt.contourf( x, z, rho_file.variables['salt'][0,:,0,:]); plt.xlim(xxl); plt.ylim(yl); plt.title(r'b) $\rho$ coordinate S');
plt.plot( xh, rho_file.variables['e'][0,:,0,:].T, 'k');
plt.subplot(223)
z = ( sigma_file.variables['e'][0,:-1,0,:] + sigma_file.variables['e'][0,1:,0,:] ) / 2
plt.contourf( x, z, sigma_file.variables['salt'][0,:,0,:]); plt.xlim(xxl); plt.ylim(yl); plt.title(r'c) $\sigma$ coordinate S');
plt.plot( xh, sigma_file.variables['e'][0,:,0,:].T, 'k');
plt.subplot(224)
z = ( z_file.variables['e'][0,:-1,0,:] + z_file.variables['e'][0,1:,0,:] ) / 2
plt.contourf( x, z, z_file.variables['salt'][0,:,0,:]); plt.xlim(xxl); plt.ylim(yl); plt.title('d) z* coordinate S');
plt.plot( xh, z_file.variables['e'][0,:,0,:].T, 'k');

def fix_contourf(nc_object, record, xh, variable='salt', clim=None, xl=None, yl=None, plot_grid=True):
    e = nc_object.variables['e'][record,:,0,:] # Interface positions
    z = ( e[:-1,:] + e[1:,:] ) / 2 # Layer centers
    S = nc_object.variables[variable][record,:,0,:] # Model output
    z = numpy.vstack( ( e[0,:], z, e[-1,:] ) ) # Add a layer at top and bottom
    S = numpy.vstack( ( S[0,:], S, S[-1,:] ) ) # Add layer data from top and bottom
    x = xh + 0*z
    plt.contourf( x, z, S );
    if clim is not None: plt.clim(clim);
    if plot_grid: plt.plot( xh, e.T, 'k');
    if xl is not None: plt.xlim(xl);
    if yl is not None: plt.ylim(yl);

plt.figure(figsize=(12,3))
# Same plot as above
plt.subplot(121)
z = ( layer_file.variables['e'][0,:-1,0,:] + layer_file.variables['e'][0,1:,0,:] ) / 2
x = xh + 0*z
plt.contourf( x, z, layer_file.variables['salt'][0,:,0,:]); plt.xlim(xxl); plt.ylim(yl);
plt.title('a) Layer mode S, as above');
plt.plot( xh, layer_file.variables['e'][0,:,0,:].T, 'k');
plt.clim(34,35)
# Now with an extra layer above and below
plt.subplot(122)
fix_contourf(layer_file, 0, xh, xl=xxl, yl=yl, clim=(34,35)); plt.title('b) Layer mode S, plotted with extra layers');

plt.figure(figsize=(12,6))
xxl=50,120 # This is the zoomed-in region around the shelf break in model coordinates
plt.subplot(221); fix_contourf(layer_file, 0, xh, xl=xxl, yl=yl, clim=(34,35)); plt.title('a) Layer mode S')
plt.subplot(222); fix_contourf(rho_file, 0, xh, xl=xxl, yl=yl, clim=(34,35)); plt.title(r'b) $\rho$ coordinate S');
plt.subplot(223); fix_contourf(sigma_file, 0, xh, xl=xxl, yl=yl, clim=(34,35)); plt.title(r'c) $\sigma$ coordinate S');
plt.subplot(224); fix_contourf(z_file, 0, xh, xl=xxl, yl=yl, clim=(34,35)); plt.title('d) z* coordinate S');

def plot_with_pcolormesh(nc_object, record, xq, variable='salt', clim=None, xl=None, yl=None, plot_grid=True):
    e = nc_object.variables['e'][record,:,0,:] # Interface positions for h-columns
    ea = numpy.vstack( ( e[:,0].T, (e[:,:-1].T+e[:,1:].T)/2, e[:,-1].T ) ).T # Interface positions averaged to u-columns
    plt.pcolormesh( xq+0*ea, ea, nc_object.variables[variable][record,:,0,:] )
    if clim is not None: plt.clim(clim);
    if plot_grid: plt.plot( xq, ea.T, 'k');
    if xl is not None: plt.xlim(xl);
    if yl is not None: plt.ylim(yl);

plt.figure(figsize=(12,6))
xxl=50,120 # This is the zoomed-in region around the shelf break in model coordinates
plt.subplot(221); plot_with_pcolormesh(layer_file, 0, xq, xl=xxl, yl=yl, clim=(34,35)); plt.title('a) Layer mode S')
plt.subplot(222); plot_with_pcolormesh(rho_file, 0, xq, xl=xxl, yl=yl, clim=(34,35)); plt.title(r'b) $\rho$ coordinate S');
plt.subplot(223); plot_with_pcolormesh(sigma_file, 0, xq, xl=xxl, yl=yl, clim=(34,35)); plt.title(r'c) $\sigma$ coordinate S');
plt.subplot(224); plot_with_pcolormesh(z_file, 0, xq, xl=xxl, yl=yl, clim=(34,35)); plt.title('d) z* coordinate S');

# These next two lines add the MOM6-examples/tools/analysis/ directory to the search path for python packages
import sys
sys.path.append('../../tools/analysis/')
# m6toolbox is a python package that has a function that helps visualize vertical sections
import m6toolbox

# Define a function to plot a section
def plot_section(file_handle, record, xq, variable='salt', clim=None, xl=None, yl=None,  plot_grid=True, rep='pcm'):
    """Plots a section by reading vertical grid and scalar variable and super-sampling
    both in order to plot vertical and horizontal reconstructions.
    
    Optional arguments have defaults for plotting salinity and overlaying the grid.
    """
    e = file_handle.variables['e'][record,:,0,:] # Vertical grid positions
    s = file_handle.variables[variable][record,:,0,:] # Scalar field to color
    x,z,q = m6toolbox.section2quadmesh(xq, e, s, representation=rep) # This yields three areas at twice the model resolution
    plt.pcolormesh(x, z, q);
    if clim is not None: plt.clim(clim)
    if plot_grid: plt.plot(x, z.T, 'k', hold=True);
    if xl is not None: plt.xlim(xl)
    if yl is not None: plt.ylim(yl)

plt.figure(figsize=(12,6))
plt.subplot(2,2,1); plot_section(layer_file, 0, xq, xl=xxl, yl=yl, clim=(34,35), rep='plm'); plt.title('a) Layer S');
plt.subplot(2,2,2); plot_section(rho_file, 0, xq, xl=xxl, yl=yl, clim=(34,35), rep='plm'); plt.title(r'b) $\rho$-coordinate S');
plt.subplot(2,2,3); plot_section(sigma_file, 0, xq, xl=xxl, yl=yl, clim=(34,35), rep='linear'); plt.title(r'c) $\sigma$-coordinate S');
plt.subplot(2,2,4); plot_section(z_file, 0, xq, xl=xxl, yl=yl, clim=(34,35), rep='pcm'); plt.title('d) z*-coordinate S');

