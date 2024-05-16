get_ipython().magic('matplotlib inline')

import pylab as plt
from filterbank import Filterbank

obs = Filterbank('voyager_f1032192_t300_v2.fil')

obs.info()

print obs.header
print obs.data.shape

obs.plot_spectrum()

obs.plot_spectrum(f_start=8420.18, f_stop=8420.25)

plt.figure(figsize=(8, 6))
plt.subplot(3,1,1)
obs.plot_spectrum(f_start=8420.193, f_stop=8420.195) # left sideband
plt.subplot(3,1,2)
obs.plot_spectrum(f_start=8420.2163, f_stop=8420.2166) # carrier
plt.subplot(3,1,3)
obs.plot_spectrum(f_start=8420.238, f_stop=8420.24) # right sideband
plt.tight_layout()

