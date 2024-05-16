get_ipython().magic('pylab inline')

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import glob

pylab.rcParams['figure.figsize'] = (16, 12)

data_dir = 'data/beatles/chordlab/The_Beatles/'

chord_files = glob.glob(data_dir + '*/*.lab.pcs.tsv')

print('total number of songs', len(chord_files))
chord_files[:5]

def read_chord_file(path):
    return pd.read_csv(path, sep='\t')

def add_track_id(df, track_id):
    df['track_id'] = track_id
    return df

def track_title(path):
    return '/'.join(path.split('/')[-2:]).replace('.lab.pcs.tsv', '')

def read_key_file(path):
    return pd.read_csv(path, sep='\t', header=None, names=['start', 'end', 'silence', 'key_label'])

selected_files = chord_files
all_chords = pd.concat(add_track_id(read_chord_file(file), track_id) for (track_id, file) in enumerate(selected_files))

all_chords['duration'] = all_chords['end'] - all_chords['start']

nonsilent_chords = all_chords[all_chords['label'] != 'N']

print('total number of chord segments', len(all_chords))

key_files = glob.glob('data/beatles/keylab/The_Beatles/*/*.lab')
len(key_files)

all_keys = pd.concat(add_track_id(read_key_file(file), track_id) for (track_id, file) in enumerate(key_files))

print('all key segments:', len(all_keys))
print('non-silence key segments:', len(all_keys['key_label'].dropna()))

all_keys['key_label'].value_counts()

all_keys['key_label'].map(lambda label: label.replace(':.*', ''))

pcs_columns = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']

def find_track(name):
    return [i for (i, path) in enumerate(chord_files) if name in path]

def draw_track(track_id):
    print(track_title(chord_files[track_id]))
    track = all_chords[all_tracks['track_id'] == track_id]
    matshow(track[pcs_columns].T)
    grid(False)
    gca().set_yticks(np.arange(12))
    gca().set_yticklabels(pcs_columns)

draw_track(find_track('Yesterday')[0])

pc_histogram = pd.DataFrame({'pitch_class': pcs_columns, 'relative_count': nonsilent_chords[pcs_columns].mean()})
stem(pc_histogram['relative_count'])
gca().set_xticks(np.arange(12))
gca().set_xticklabels(pcs_columns);

pc_histogram.sort('relative_count', ascending=False, inplace=True)

plot(pc_histogram['relative_count'],'o:')
gca().set_xticks(np.arange(12))
gca().set_xticklabels(pc_histogram['pitch_class']);
ylim(0, 1);
xlim(-.1, 11.1);

chord_histogram = all_chords['label'].value_counts()

chord_histogram

print('number of unique chords (including silence):', len(chord_histogram))

plot(chord_histogram);

chord_root_histogram = nonsilent_chords['root'].value_counts()
# convert index from integers to symbolic names
chord_root_histogram.index = pd.Series(pcs_columns)[chord_root_histogram.index].values
chord_root_histogram

#all_chords[pcs_columns + ['track_id']]

all_chords

duration = all_chords['duration']
duration.hist(bins=100);

sns.distplot(duration[duration < 10], bins=100)
xlabel('duration (sec)');

X = all_chords[['duration'] + pcs_columns].astype(np.float32)
y = all_chords['track_id'].astype(np.int32)

from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

len(X_train), len(X_valid), len(X_test)

from sklearn.linear_model import LogisticRegression

lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)

from sklearn.metrics import classification_report, confusion_matrix

y_pred_lr = lr_model.predict(X_valid)

lr_model.score(X_valid, y_valid)

print(classification_report(y_valid, y_pred_lr))

matshow(confusion_matrix(y_valid, y_pred_lr), cmap=cm.Spectral_r)
colorbar();

import theanets
import climate # some utilities for command line interfaces
climate.enable_default_logging()

exp = theanets.Experiment(
    theanets.Classifier,
    layers=(13, 50, 180),
    hidden_l1=0.1)

exp.train(
    (X_train, y_train),
    (X_valid, y_valid),
    optimize='rmsprop',
    learning_rate=0.01,
    momentum=0.5)

y_pred_nn = exp.network.classify(X_valid)
y_pred_nn

print(classification_report(y_valid, y_pred_nn))

matshow(confusion_matrix(y_valid, y_pred_nn), cmap=cm.Spectral_r)
colorbar();





