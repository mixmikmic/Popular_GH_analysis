model_file = '../data/models/pretrained_model.th'

get_ipython().system('cat sad_eg/rs13336428.vcf')

import subprocess

targets_file = 'sad_eg/sample_beds.txt'

cmd = 'basset_sad.py -l 600 -i -o sad -s -t %s %s sad_eg/rs13336428.vcf' % (targets_file, model_file)
subprocess.call(cmd, shell=True)

# actual file is sad/sad_Bone_mineral-rs13336428-A_heat.pdf

from IPython.display import Image
Image(filename='sad_eg/sad_Bone_mineral-rs13336428-A_heat.png')

get_ipython().system(' sort -k9 -g -r  sad/sad_table.txt | head')

cmd = 'basset_sat_vcf.py -t 6 -o sad/sat %s sad_eg/rs13336428.vcf' % model_file 
subprocess.call(cmd, shell=True)

# actual file is sad/sat/rs13336428_G_c6_heat.pdf

Image(filename='sad_eg/rs13336428_G_c6_heat.png')

# actual file is sad/sat/rs13336428_A_c6_heat.pdf

Image(filename='sad_eg/rs13336428_A_c6_heat.png')

