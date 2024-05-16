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
#  Beware, for MATH display, use %%latex, NOT the following:
#                   from IPython.display import Math
#                   from IPython.display import Latex
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

#  First download latest GOLD reports:
cotr = cotr_get( f4xau )
#                ^ = 'GC' currently.

#  Show column labels and only the last report:
tail( cotr, 1 )

longs  = cotr['Money Manager Longs']
shorts = cotr['Money Manager Shorts']

# difference in number of contracts:
lsdiff = todf( longs - shorts )

plot( lsdiff )

#  Scale-free measure from zero to 1:
z_xau = todf( longs / (longs + shorts ))

#  Could interpret as prob( bullish Manager ):
plot( z_xau )

# How does our indicator compare to spot gold prices?
xau = get( d4xau )
plot( xau['2006-06-12':] )
# using the FRED database for Gold London PM fix.

#       ? or ?? is useful to investigate source.
get_ipython().magic('pinfo2 cotr_position')

#  Silver position:
z_xag = cotr_position( f4xag )
plot( z_xag )

#  Correlation and regression:
#  stat2( z_xag['Y'], z_xau['Y'] )

#  PRECIOUS METALS position:
z_metals = get( w4cotr_metals )
plot( z_metals )

z_metal_ema = ema(z_metals, 0.05)
plot( z_metal_ema )

#  Dollar position (not price):
z_usd = get( w4cotr_usd )
plot( z_usd )

# Fed US Dollar index, m4 means monthly frequency:
usd = get( m4usdrtb )
plot( usd['2006-06-01':] )

#  Bonds position:
z_bonds = get( w4cotr_bonds )
plot( z_bonds )

bondrate = get( d4bond10 )
#  10-y Treasury rate INVERTED in lieu of price:
plot( -bondrate['2006-06-01':] )

#  Equities position:
z_eq = get( w4cotr_equities )
plot( z_eq  )

#  So let's normalize the equities indicator:
plot(normalize( z_eq ))

#  SPX price data showing the post-2009 equities bull market:
spx = get( d4spx )
plot( spx['2006-06-12':] )

# class consists of precious metals, US dollar, bonds, equities:
z_class = [ z_metals, z_usd, z_bonds, z_eq ]

z_data = paste( z_class )
z_data.columns = ['z_metals', 'z_usd', 'z_bonds', 'z_eq']
#  Created "group" dataframe z_data with above named columns.
#  Please see fecon235 module for more details.

stats( z_data )

#  Normalize indicators:
z_nor = groupfun( normalize, z_data )

#  Compare recent normalized indicators:
tail( z_nor, 24 )

#  Asset classes are specified as a "group" in a fecon235 dictionary:
cotr4w

#  Source code for retrieval of COTR [0,1] position indicators
#  (exactly like z_data, except column names are nicer),  
#  followed by operator to normalize the position indicators:
get_ipython().magic('pinfo2 groupcotr')

#  This encapsulates the TECHNIQUES in this notebook,
#  including the option to apply smoothing.

#  group* functions are defined in the fecon235 module.

#  MAIN in action!
norpos = groupcotr( cotr4w, alpha=0 )

#  Most recent results as a test exhibit:
tail( norpos, 3 )

#  To broadly visualize a decade of data, 
#  we resample the weekly normalized position indicators
#  into MONTHLY frequency (median method).
norpos_month = groupfun( monthly, norpos )

#  Plot from 2006 to present:
scatter( norpos_month, col=[0, 1] )

