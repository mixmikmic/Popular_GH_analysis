from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from builtins import *
from io import open

import time, datetime

from NoisyNLP.utils import *
from NoisyNLP.features import *
from NoisyNLP.models import *
from NoisyNLP.experiments import *

train_files = ["./data/cleaned/train.BIEOU.tsv"]
dev_files = ["./data/cleaned/dev.BIEOU.tsv", "./data/cleaned/dev_2015.BIEOU.tsv"]
test_files = ["./data/cleaned/test.BIEOU.tsv"]
vocab_file = "./vocab.no_extras.txt"
outdir = "./test_exp"
wordvec_file = "/home/entity/Downloads/GloVe/glove.twitter.27B.200d.txt.processed.txt"
dictionary_dir="./data/cleaned/custom_lexicons/"
gimple_twitter_brown_clusters_dir="/home/entity/Downloads/GloVe/50mpaths2"
data_brown_cluster_dir="word_clusters/"
data_clark_cluster_dir="clark_clusters/"

exp = Experiment(outdir, train_files, dev_files, test_files, vocab_file)
all_sequences = [[t[0] for t in seq] 
                        for seq in (exp.train_sequences + exp.dev_sequences + exp.test_sequences)]
print("Total sequences: ", len(all_sequences))

wv_model = WordVectors(all_sequences,wordvec_file)
word2vec_clusters = wv_model.get_clusters(n_clusters=50)

dict_features = DictionaryFeatures(dictionary_dir)

gimple_brown_cf = ClusterFeatures(gimple_twitter_brown_clusters_dir, cluster_type="brown")
gimple_brown_cf.set_cluster_file_path(gimple_twitter_brown_clusters_dir)
gimple_brown_clusters = gimple_brown_cf.read_clusters()

data_brown_cf = ClusterFeatures(data_brown_cluster_dir, cluster_type="brown")
data_brown_cf.set_cluster_file_path()
data_brown_clusters = data_brown_cf.read_clusters()

data_clark_cf = ClusterFeatures(data_clark_cluster_dir, cluster_type="clark", n_clusters=32)
data_clark_cf.set_cluster_file_path()
data_clark_clusters = data_clark_cf.read_clusters()

import pickle

with open("pickled_data/wv_model.pkl", "wb+") as fp:
    pickle.dump(wv_model, fp)

with open("pickled_data/word2vec_clusters.pkl", "wb+") as fp:
    pickle.dump(word2vec_clusters, fp)

print("Done")

def get_X_y_exp1(sequences):
    X = [sent2features(s, vocab=None,
                         dict_features=dict_features, vocab_presence_only=False,
                         window=2, interactions=True, dict_interactions=False,
                         lowercase=True, dropout=0, word2vec_model=wv_model.model,
                        cluster_vocabs=[
            gimple_brown_clusters,
            data_brown_clusters,
            data_clark_clusters
        ])
         for s in sequences]
    y = [sent2labels(s) for s in sequences]
    return X, y

exp.gen_model_data(proc_func=get_X_y_exp1)

exp.fit_evaluate()

exp.describe_model()

outdir = "./test_exp_train_dev"
exp = Experiment(outdir, train_files + dev_files, dev_files, test_files, vocab_file)

exp.gen_model_data(proc_func=get_X_y_exp1)

exp.fit_evaluate()

exp.describe_model()

test_enriched_data_brown_cluster_dir="test_enriched/brown_clusters/"
test_enriched_data_clark_cluster_dir="test_enriched/clark_clusters/"

brown_exec_path="/home/entity/Downloads/brown-cluster/wcluster"
brown_input_data_path="test_enriched/all_sequences.brown.txt"
test_enriched_data_brown_cf = ClusterFeatures(test_enriched_data_brown_cluster_dir,
                                              cluster_type="brown", n_clusters=100)

test_enriched_data_brown_cf.set_cluster_file_path()
test_enriched_data_brown_clusters = test_enriched_data_brown_cf.read_clusters()

test_enriched_data_clark_cf = ClusterFeatures(test_enriched_data_clark_cluster_dir,
                                              cluster_type="clark", n_clusters=32)
test_enriched_data_clark_cf.set_cluster_file_path()
test_enriched_data_clark_clusters = test_enriched_data_clark_cf.read_clusters()

def get_X_y_exp2(sequences):
    X = [sent2features(s, vocab=None,
                         dict_features=dict_features, vocab_presence_only=False,
                         window=2, interactions=True, dict_interactions=False,
                         lowercase=True, dropout=0, word2vec_model=wv_model.model,
                        cluster_vocabs=[
            gimple_brown_clusters,
            test_enriched_data_brown_clusters,
            test_enriched_data_clark_clusters
        ])
         for s in sequences]
    y = [sent2labels(s) for s in sequences]
    return X, y

outdir = "./test_exp_train_dev_test_enriched"
exp = Experiment(outdir, train_files + dev_files, dev_files, test_files, vocab_file)

exp.gen_model_data(proc_func=get_X_y_exp2)

exp.fit_evaluate()

def get_X_y_exp3(sequences):
    X = [sent2features(s, vocab=None,
                         dict_features=dict_features, vocab_presence_only=False,
                         window=4, interactions=True, dict_interactions=True,
                         lowercase=True, dropout=0, word2vec_model=wv_model.model,
                        cluster_vocabs=[
            gimple_brown_clusters,
            test_enriched_data_brown_clusters,
            test_enriched_data_clark_clusters
        ])
         for s in sequences]
    y = [sent2labels(s) for s in sequences]
    return X, y

outdir = "./test_exp_tdt_enriched_win4_dict_intract"
exp = Experiment(outdir, train_files + dev_files, dev_files, test_files, vocab_file)

exp.gen_model_data(proc_func=get_X_y_exp3)

exp.fit_evaluate()

def get_X_y_exp4(sequences):
    X = [sent2features(s, vocab=None,
                         dict_features=dict_features, vocab_presence_only=False,
                         window=4, interactions=True, dict_interactions=True,
                         lowercase=False, dropout=0, word2vec_model=wv_model.model,
                        cluster_vocabs=[
            gimple_brown_clusters,
            test_enriched_data_brown_clusters,
            test_enriched_data_clark_clusters
        ])
         for s in sequences]
    y = [sent2labels(s) for s in sequences]
    return X, y

outdir = "./test_exp_tdt_enriched_word_Features"
exp = Experiment(outdir, train_files + dev_files, dev_files, test_files, vocab_file)

exp.gen_model_data(proc_func=get_X_y_exp4)

exp.fit_evaluate()

get_ipython().magic('load_ext autoreload')

get_ipython().magic('aimport NoisyNLP.experiments')

get_ipython().magic('autoreload 2')

outdir = "./test_exp_tdt_enriched_word_Features_global"
exp = Experiment(outdir, train_files + dev_files, dev_files, test_files, vocab_file)
cat_names = list(set([t[1][2:] for seq in exp.train_sequences for t in seq if t[1] != "O"]))
cat_names

global_features = GlobalFeatures(word2vec_model=wv_model.model, cluster_vocabs=gimple_brown_clusters, cat_names=cat_names)

exp.train_sequences[:1]

global_features.fit_model(exp.train_sequences, test_sequences=exp.dev_sequences)

def get_X_y_exp5(sequences):
    preds = global_features.get_global_predictions(sequences)
    X = [sent2features(s, vocab=None,
                         dict_features=dict_features, vocab_presence_only=False,
                         window=4, interactions=True, dict_interactions=True,
                         lowercase=False, dropout=0, word2vec_model=wv_model.model,
                        cluster_vocabs=[
            gimple_brown_clusters,
            test_enriched_data_brown_clusters,
            test_enriched_data_clark_clusters
        ],
                       extra_features=global_features.get_global_sequence_features(s, predictions=p))
         for s,p in zip(sequences, preds)]
    y = [sent2labels(s) for s in sequences]
    return X, y

exp.gen_model_data(proc_func=get_X_y_exp5)

exp.fit_evaluate()



