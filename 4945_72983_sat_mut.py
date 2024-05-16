model_file = '../data/models/pretrained_model.th'
seqs_file = '../data/encode_roadmap.h5'

import subprocess

cmd = 'basset_sat.py -t 46 -n 200 -s 10 -o satmut %s %s' % (model_file, seqs_file)
subprocess.call(cmd, shell=True)

# actual file is satmut/chr17_4904020-4904620\(+\)_c46_heat.pdf

from IPython.display import Image
Image(filename='satmut_eg/chr17_4904020-4904620(+)_c46_heat.png')

cmd = 'basset_sat.py -t -1 -n 200 -o satmut_hox %s satmut_eg/hoxa_boundary.fa' % model_file
subprocess.call(cmd, shell=True)

# actual file is satmut_hox/chr7_27183235-27183835_c127_heat.pdf

Image(filename='satmut_eg/chr7_27183235-27183835_c127_heat.png')

