from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from builtins import *
from io import open

import time, datetime

from NoisyNLP.utils import *
from NoisyNLP.features import *
from NoisyNLP.models import *
from NoisyNLP.experiments import *

import pickle

train_files = ["./data/cleaned/train.BIEOU.tsv"]
dev_files = ["./data/cleaned/dev.BIEOU.tsv", "./data/cleaned/dev_2015.BIEOU.tsv"]
test_files = ["./data/cleaned/test.BIEOU.tsv"]
vocab_file = "./vocab.no_extras.txt"
outdir = "./test_exp"
test_enriched_data_brown_cluster_dir="test_enriched/brown_clusters/"
test_enriched_data_clark_cluster_dir="test_enriched/clark_clusters/"

exp = Experiment(outdir, train_files, dev_files, test_files, vocab_file)
all_sequences = [[preprocess_token(t[0], to_lower=True) for t in seq] 
                        for seq in (exp.train_sequences + exp.dev_sequences + exp.test_sequences)]
print("Total sequences: ", len(all_sequences))

brown_exec_path="/home/entity/Downloads/brown-cluster/wcluster"
brown_input_data_path="test_enriched/all_sequences.brown.txt"
test_enriched_data_brown_cf = ClusterFeatures(test_enriched_data_brown_cluster_dir,
                                              cluster_type="brown", n_clusters=100)
test_enriched_data_brown_cf.set_exec_path(brown_exec_path)
test_enriched_data_brown_cf.gen_training_data(all_sequences, brown_input_data_path)

test_enriched_data_brown_cf.gen_clusters(brown_input_data_path, test_enriched_data_brown_cluster_dir)

clark_exec_path="/home/entity/Downloads/clark_pos_induction/src/bin/cluster_neyessenmorph"
clark_input_data_path="test_enriched/all_sequences.clark.txt"
test_enriched_data_clark_cf = ClusterFeatures(test_enriched_data_clark_cluster_dir,
                                              cluster_type="clark", n_clusters=32)
test_enriched_data_clark_cf.set_exec_path(clark_exec_path)
test_enriched_data_clark_cf.gen_training_data(all_sequences, clark_input_data_path)

test_enriched_data_clark_cf.gen_clusters(clark_input_data_path, test_enriched_data_clark_cluster_dir)



