get_ipython().magic('pylab inline')
from pylab import *
import codecs,string,os,sys,os.path,glob,re

get_ipython().system('test -f uw3-500.tgz || wget -nd http://www.tmbdev.net/ocrdata/uw3-500.tgz')

get_ipython().system('test -d book || tar -zxvf uw3-500.tgz')

get_ipython().system('ls book/0005/010001.*')

get_ipython().system('dewarp=center report_every=500 save_name=test save_every=10000 ntrain=11000 ../clstmctc uw3-500.h5')

get_ipython().system('ls book/*/*.bin.png | sort -r > uw3.files')
get_ipython().system('sed 100q uw3.files > uw3-test.files')
get_ipython().system('sed 1,100d uw3.files > uw3-train.files')
get_ipython().system('wc -l uw3*.files')

get_ipython().system('params=1 save_name=uw3small save_every=1000 report_every=100 maxtrain=50000 test_every=1000 ../clstmocrtrain uw3-train.files uw3-test.files')



