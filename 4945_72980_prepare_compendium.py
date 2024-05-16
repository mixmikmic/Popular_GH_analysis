get_ipython().system('cd ../data; preprocess_features.py -y -m 200 -s 600 -o er -c genomes/human.hg19.genome sample_beds.txt')

get_ipython().system('bedtools getfasta -fi ../data/genomes/hg19.fa -bed ../data/er.bed -s -fo ../data/er.fa')

get_ipython().system('seq_hdf5.py -c -r -t 71886 -v 70000 ../data/er.fa ../data/er_act.txt ../data/er.h5')

