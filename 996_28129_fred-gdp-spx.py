from fecon235.fecon235 import *

#  PREAMBLE-p6.15.1223 :: Settings and system details
from __future__ import absolute_import, print_function
system.specs()
pwd = system.getpwd()   # present working directory as variable.
print(" ::  $pwd:", pwd)
#  If a module is modified, automatically reload it:
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')
#       Use 0 to disable this feature.

#  Notebook DISPLAY options:
#      Represent pandas DataFrames as text; not HTML representation:
import pandas as pd
pd.set_option( 'display.notebook_repr_html', False )
from IPython.display import HTML # useful for snippets
#  e.g. HTML('<iframe src=http://en.mobile.wikipedia.org/?useformat=mobile width=700 height=350></iframe>')
from IPython.display import Image 
#  e.g. Image(filename='holt-winters-equations.png', embed=True) # url= also works
from IPython.display import YouTubeVideo
#  e.g. YouTubeVideo('1j_HxD4iLn8', start='43', width=600, height=400)
from IPython.core import page
get_ipython().set_hook('show_in_pager', page.as_hook(page.display_page), 0)
#  Or equivalently in config file: "InteractiveShell.display_page = True", 
#  which will display results in secondary notebook pager frame in a cell.

#  Generate PLOTS inside notebook, "inline" generates static png:
get_ipython().magic('matplotlib inline')
#          "notebook" argument allows interactive zoom and resize.

#  Note that y represents the raw series in the equations.
#  We use Y as a generic label in dataframes.

Image(filename='holt-winters-equations.png', embed=True)

#  Retrieve quarterly data for real US GDP as DataFrame:
dfy = get( q4gdpusr )

#  So we are currently discussing a 17 trillion dollar economy...
tail( dfy )

#  DEFINE start YEAR OF ANALYSIS:
start = '1980'

#  We can easily re-run the entire notebook from different start date.

print(hw_alpha, hw_beta, " :: DEFAULT Holt-Winters alpha and beta")

#  This optimization procedure may be computationally intensive
#  depending on data size and the number of grids,
#  so uncomment the next command to run it:

ab = optimize_holt( dfy['1980':], grids=50 )

ab

#  Let's use the optimized values for alpha and beta
#  to compute the Holt dataframe:

holtdf = holt( dfy, alpha=ab[0], beta=ab[1] )

stats( holtdf[start:] )
#  Summary stats from custom start point:

#  Y here is the raw series, i.e. real GDP in billions of dollars:
plot( holtdf['Y'][start:] )

#  Annualized geometric mean return of the raw series:
georet( todf(holtdf['Y'][start:]), yearly=4 )

#  Since 1980, real GDP growth is about 2.6%

#  Level can be thought of as the smoothed series:

# plot( holtdf['Level'][start:] )

#  Growth is the fitted slope at each point,
#  expressed in units of the original series:

# plot( holtdf['Growth'][start:] )

pc = holtpc( dfy, yearly=4, alpha=ab[0], beta=ab[1] )
plot( pc[start:] )

gdp_forecast = tailvalue( pc )
gdp_forecast

#  Here is the BIG GDP TREND since the end of World War 2:

trendpc = trend( pc )

plot( trendpc )

stat( trendpc )

detpc = detrend( pc )
plot( detpc )

stat( detpc )

#  Forecast real GDP, four quarters ahead:
foregdp = holtforecast( holtdf, h=4 )
foregdp

#  Forecast real GDP rate of GROWTH, four quarters ahead:
100 * ((foregdp.iloc[4] / foregdp.iloc[0]) - 1)

#  We can plot the point forecasts 12 quarters ahead (i.e. 3 years):

# plotholt( holtdf, 12 )

#  SPX is a daily series, but we can directly retrieve its monthly version:
spx = get( m4spx )

#  ... to match the monthly periodicity of our custom deflator:
defl = get( m4defl )

#  Now we synthesize a quarterly real SPX version by resampling:
spdefl = todf( spx * defl )
spq = quarterly( spdefl )

#  Real SPX resampled quarterly in current dollars:
plot( spq )

#  Geometric mean return for real SPX:
georet( spq[start:], yearly=4 )

#  cf. volatility for real GDP = 1.5% 
#      in contrast to equities = 13.2%

stat2( dfy['Y'][start:], spq['Y'][start:] )

#  2017-04-09, for start='1980':  5.0 * (s/g) = 

gsratio = 5.0 * div(spq[start:].mean(), dfy[start:].mean())
gsratio = gsratio.values[0]
gsratio

#       holtpc should be familiar from our GDP analysis.
pcspq = holtpc( spq, yearly=4, alpha=hw_alpha, beta=hw_beta )
plot( pcspq )

#  Note we use all data since 1957.

spx_forecast = tailvalue( pcspq )
spx_forecast

trend_pcspq = trend( pcspq )
plot( trend_pcspq )

det_pcspq = detrend( pcspq )
plot( det_pcspq )

