#  Call the MAIN module: 
from fecon235.fecon235 import *
#  This loose import style is acceptable only within 
#  interactive environments outside of any fecon235 packages.
#  (Presence of __init__.py in a directory indicates 
#  it is a "package.") 
#
#  These directories: nb and tests, are explicitly NOT packages.

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

#  What the heck is "system" mentioned in the preamble?
get_ipython().magic('pinfo system')

#  Assign a name to a dataframe
#  that will contain monthly unemployment rates.

unem = get( m4unemp )
#           m4 implies monthly frequency.

#  But does m4unemp really represent?
get_ipython().magic('pinfo m4unemp')

#  Illustrate slicing: 1997 <= unem <= 2007:
unem07 = unem['1997':'2007']
#  Verify below by Head and Tail.

#  Quick summary:
stat( unem07 )

#  More verbose statistical summary:
stats( unem07 )

get_ipython().magic('pinfo2 stats')

# #  Uncomment to see how numpy computes something simple as absolute value:
# np.abs??

