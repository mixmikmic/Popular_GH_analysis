get_ipython().run_cell_magic('bash', '', 'python preprocessing.py --help')

get_ipython().run_cell_magic('bash', '', 'head -n 5 corpus/corpus.raw')

get_ipython().run_cell_magic('bash', '', 'python preprocessing.py -p corpus/corpus.raw corpus/corpus.nopunct')

get_ipython().run_cell_magic('bash', '', 'head -n 5 corpus/corpus.nopunct')

get_ipython().run_cell_magic('bash', '', 'python preprocessing.py -ps corpus/corpus.raw corpus/corpus.nostop')

get_ipython().run_cell_magic('bash', '', 'head -n 5 corpus/corpus.nostop')

get_ipython().run_cell_magic('bash', '', 'python preprocessing.py -psu corpus/corpus.raw corpus/corpus.nouml')

get_ipython().run_cell_magic('bash', '', 'head -n 5 corpus/corpus.nouml')

get_ipython().run_cell_magic('bash', '', 'python preprocessing.py -psub corpus/corpus.raw corpus/corpus.psu')

get_ipython().run_cell_magic('bash', '', 'head -n 5 corpus/corpus.psu.bigram')

