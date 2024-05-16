model_file = '../data/models/pretrained_model.th'
seqs_file = '../data/encode_roadmap.h5'

import subprocess

cmd = 'basset_test.lua %s %s test_out' % (model_file, seqs_file)
subprocess.call(cmd, shell=True)

get_ipython().system('head test_eg/aucs.txt')

targets_file = '../data/sample_beds.txt'

cmd = 'plot_roc.py -t %s test_out' % (targets_file)
subprocess.call(cmd, shell=True)

# actual file is test_out/roc1.pdf

from IPython.display import Image
Image(filename='test_eg/roc1.png')

