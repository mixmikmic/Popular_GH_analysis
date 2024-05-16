import numpy
import scipy.io.netcdf
import matplotlib.pyplot as plt
get_ipython().magic('pylab inline')
pylab.rcParams['figure.figsize'] = (16.0, 4.0)

bml_prog_z=scipy.io.netcdf_file('BML/prog_z.nc','r')
kpp_prog_z=scipy.io.netcdf_file('KPP/prog_z.nc','r')
epbl_prog_z=scipy.io.netcdf_file('EPBL/prog_z.nc','r')
bml_visc=scipy.io.netcdf_file('BML/visc.nc','r')
kpp_visc=scipy.io.netcdf_file('KPP/visc.nc','r')
epbl_visc=scipy.io.netcdf_file('EPBL/visc.nc','r')

t = bml_prog_z.variables['Time'][:]
zw = -bml_prog_z.variables['zw'][:]
zt = -bml_prog_z.variables['zt'][:]

plt.subplot(131);
plt.contourf(t[1:], zt[:19], bml_prog_z.variables['temp'][1:,:19,1,1].T, levels=numpy.arange(14.4,15.06,.02));
plt.colorbar(); plt.xlabel('Time (days)'); plt.ylabel('z* (m)'); plt.title(r'BML $\theta(z,t)$');
plt.subplot(132);
plt.contourf(t[1:], zt[:19], kpp_prog_z.variables['temp'][1:,:19,1,1].T, levels=numpy.arange(14.4,15.06,.02));
plt.colorbar(); plt.xlabel('Time (days)'); plt.ylabel('z* (m)'); plt.title(r'KPP $\theta(z,t)$');
plt.subplot(133);
plt.contourf(t[1:], zt[:19], epbl_prog_z.variables['temp'][1:,:19,1,1].T, levels=numpy.arange(14.4,15.06,.02));
plt.colorbar(); plt.xlabel('Time (days)'); plt.ylabel('z* (m)'); plt.title(r'EPBL $\theta(z,t)$');

rho0 = 1000.; dRhodT=-0.255; dTdz= 0.01; g=9.80616; N2= -g/rho0*dRhodT*dTdz; print('N2 =',N2)
ustar = sqrt(0.05/rho0); print('u* = ',ustar)
mstar=0.3
plt.subplot(121);
plt.plot(t[1:], bml_prog_z.variables['temp'][1:,0,1,1].T, label='BML');
plt.plot(t[1:], kpp_prog_z.variables['temp'][1:,0,1,1].T, label='KPP');
plt.plot(t[1:], epbl_prog_z.variables['temp'][1:,0,1,1].T, label='EPBL');
plt.legend(loc='lower left'); plt.xlabel('Time (days)'); plt.ylabel('SST ($\degree$C)');
plt.subplot(122);
plt.plot(t[:], bml_visc.variables['MLD_003'][:,1,1].T, label='BML');
plt.plot(t[:], kpp_visc.variables['MLD_003'][:,1,1].T, label='KPP');
plt.plot(t[:], epbl_visc.variables['MLD_003'][:,1,1].T, label='EPBL');
plt.plot(t, (12.*(ustar**3.)*mstar*(86400.*t)/N2)**(1./3.), label='Niiler and Krauss, 1977');
plt.legend(loc='upper left'); plt.xlabel('Time (days)'); plt.ylabel('MLD$_{0.03}$ (m)');

