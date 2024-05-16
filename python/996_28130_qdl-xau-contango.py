from fecon235.fecon235 import *

#  PREAMBLE-p6.16.0428 :: Settings and system details
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

#  SET UP the particular (f4) futures contracts of interest:
s_libor = 'f4libor16z'
s_xau1  = 'f4xau16z'
s_xau2  = 'f4xau17m'

#  f4libor* refers to the CME Eurodollar futures.

#  The second nearby contract for gold (xau)
#  should be 6 months after the first using the 
#  June (m) and December (z) cycle

#  RE-RUN this entire study by merely changing the string symbols.

#  Retrieve data:
libor = todf( 100 - get(s_libor) )
#             ^convert quotes to conventional % format
xau1 = get(s_xau1)
xau2 = get(s_xau2)

tail(libor)

tail(xau1)

tail(xau2)

#  Compute the contango in terms of annualized percentage:
contango = todf( ((xau2 / xau1) - 1) * 200 )

#  Multiply by 200 instead of 100 since 
#  the gold contracts are stipulated to be six months apart.

tail( contango )

plot( contango )

tango = todf( contango - libor )

tail( tango )

#  MAIN chart <- pay attention here !!
plot( tango )

tango.describe()

#  For historians:
Image(url='https://www.mcoscillator.com/data/charts/weekly/GOFO_1mo_1995-2014.gif', embed=False)

xau = get( d4xau )

#  This matches the futures sample size:
xau0 = tail( xau, 512 )

plot( xau0 )

#  Is there near-term correlation between price and tango?
#  stat2( xau0[Y], tango[Y] )
#  2015-09-11  correlation: 0.09, so NO.

#  Running annual percentage change in spot price:
xau0pc = tail( pcent(xau, 256), 512 )

plot ( xau0pc )

#  Is there near-term correlation between price change and tango?
stat2( xau0pc[Y], tango[Y] )
#  2015-09-11  correlation: -0.85, so YES.
#  2015-10-09  correlation: -0.83
#  2016-12-02  correlation: +0.81, but change in sign!

