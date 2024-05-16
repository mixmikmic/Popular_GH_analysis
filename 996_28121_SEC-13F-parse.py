from fecon235.fecon235 import *

#  pandas will give best results
#  if it can call the Python package: lxml,
#  and as a fallback: bs4 and html5lib.
#  They parse (non-strict) XML and HTML pages.
#  Be sure those three packages are pre-installed.

from fecon235.lib import yi_secform
#  We are going to derive this module in this notebook.

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

#            https cannot be read by lxml, surprisingly.
druck150814='http://www.sec.gov/Archives/edgar/data/1536411/000153641115000006/xslForm13F_X01/form13f_20150630.xml'

#     START HERE with a particular URL:
url = druck150814

#  Let's display the web page as in the browser to understand the semantics:
HTML("<iframe src=" + url + " width=1400 height=350></iframe>")

#  Use pandas to read in the xml page...
#  See http://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_html.html

#  It searches for <table> elements and only for <tr> and <th> rows and <td> elements 
#  within each <tr> or <th> element in the table.

page = pd.read_html( url )

#  Nasty output in full:

#uncomment:  page

#  page is a list of length 4:
len( page )

#  But only the last element of page interests us:
df = page[-1]
#  which turns out to be a dataframe!

#  Let's rename columns for our sanity:
df.columns = [ 'stock', 'class', 'cusip', 'usd', 'size', 'sh_prin', 'putcall', 'discret', 'manager', 'vote1', 'vote2', 'vote3'] 

#  But first three rows are SEC labels, not data, 
#  so delete them:
df = df[3:]

#  Start a new index from 0 instead of 3:
df.reset_index( drop=True )

#  Delete irrevelant columns:
dflite = df.drop( df.columns[[1, 4, 5, 7, 8, 9, 10, 11]], axis=1 )
#         inplac=True only after pandas 0.13
#uncomment: dflite

#  usd needs float type since usd was read as string:
dflite[['usd']] = dflite[['usd']].astype( float )
#                  Gotcha: int as type will fail for NaN

#  Type change allows proper sort:
dfusd = dflite.sort_values( by=['usd'], ascending=[False] )

usdsum = sum( dfusd.usd )
#  Portfolio total in USD:
usdsum

#  New column for percentage of total portfolio:
dfusd['pcent'] = np.round(( dfusd.usd / usdsum ) * 100, 2)

#  Top 20 Hits!
dfusd.head( 20 )

get_ipython().magic('pinfo2 yi_secform.pcent13f')

yi_secform.pcent13f( druck150814, 20 )

#  Simply enter the Information Table html URL for a 13F filing, 
#  and bang... [verifying our output in the previous cell]:

#  13F for Paulson & Co. filed 2015-08-14:
paulson150814 = 'http://www.sec.gov/Archives/edgar/data/1035674/000114036115032242/xslForm13F_X01/form13fInfoTable.xml'

yi_secform.pcent13f( paulson150814, 20 )

druck151113 = 'http://www.sec.gov/Archives/edgar/data/1536411/000153641115000008/xslForm13F_X01/form13f_20150930.xml'
paulson151116 = 'http://www.sec.gov/Archives/edgar/data/1035674/000114036115041689/xslForm13F_X01/form13fInfoTable.xml'

# Druckenmiller 13F for 2015-11-13:
yi_secform.pcent13f( druck151113, 20 )

#  Paulson 13F for 2015-11-16:
yi_secform.pcent13f( paulson151116, 20 )

druck160216='http://www.sec.gov/Archives/edgar/data/1536411/000153641116000010/xslForm13F_X01/form13f_20151231.xml'
paulson160216='http://www.sec.gov/Archives/edgar/data/1035674/000114036116053318/xslForm13F_X01/form13fInfoTable.xml'

# Druckenmiller 13F for 2016-02-16:
yi_secform.pcent13f( druck160216, 20 )

#  Paulson 13F for 2016-02-16:
yi_secform.pcent13f( paulson160216, 20 )

