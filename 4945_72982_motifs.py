model_file = '../data/models/pretrained_model.th'
seqs_file = '../data/encode_roadmap.h5'

import subprocess

cmd = 'basset_motifs.py -s 1000 -t -o motifs_out %s %s' % (model_file, seqs_file)
subprocess.call(cmd, shell=True)

# actual file is motifs_out/filter9_heat.pdf

from IPython.display import Image
Image(filename='motifs_eg/filter9_heat.png') 

# actual file is motifs_out/filter9_logo.eps

Image(filename='motifs_eg/filter9_logo.png') 

get_ipython().system('sort -k6 -gr motifs_out/table.txt | head -n20')

get_ipython().system('open motifs_out/tomtom/tomtom.html')

cmd = 'basset_motifs_infl.py'
cmd += ' -m motifs_out/table.txt'
cmd += ' -s 2000 -b 500'
cmd += ' -o infl_out'
cmd += ' --subset motifs_eg/primary_cells.txt'
cmd += ' -t motifs_eg/cell_activity.txt'
cmd += ' --width 7 --height 40 --font 0.5'
cmd += ' %s %s' % (model_file, seqs_file)

subprocess.call(cmd, shell=True)

