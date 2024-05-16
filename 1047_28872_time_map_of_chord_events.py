get_ipython().magic('pylab inline')

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

pylab.rcParams['figure.figsize'] = (16, 12)

data_dir = 'data/beatles/chordlab/The_Beatles/'

file = "03_-_A_Hard_Day's_Night/03_-_If_I_Fell.lab"
def read_chord_file(path):
    return pd.read_csv(path, sep=' ', header=None, names=['start','end','chord'])
df = read_chord_file(data_dir + file)

df.head()

plot(df['start'], 'g.', label='start')
plot(df['end'], 'r.', label='end')
legend(loc='center right');

print('event count:', len(df))
print('total time:', df['end'].iloc[-1])
print('last start event time:', df['start'].iloc[-1])

df['duration'] = df['end'] - df['start']
df['duration'].describe()

plot(df['duration'], '.')
xlabel('segment index')
ylabel('duration');

sns.distplot(df['duration'], axlabel='duration (sec)', rug=True, kde=False, bins=20)
title('Duration of chord segments');

def add_differences(df, col='start'):
    df['prev'] = df[col].diff(1)
    df['next'] = -df[col].diff(-1)
    return df
df_diff = add_differences(df).dropna()

df_diff.head()

sns.distplot(df_diff['prev'], label='time to previous', rug=True, kde=False, bins=10)
sns.distplot(df_diff['next'], label='time to next', rug=True, kde=False, bins=10)
legend();

def plot_time_map(df_diff, coloring=None):
    cmap = plt.cm.get_cmap('RdYlBu')
    c = np.linspace(0, 1, len(df_diff)) if coloring is None else coloring
    scatter(df_diff['prev'], df_diff['next'],
            alpha=0.5,
            c=c,
            cmap=cmap,
            edgecolors='none')
    xlabel('time to previous event')
    ylabel('time to next event')
    title('Time map')
    axes().set_aspect('equal')
    
    max_value = df_diff[['prev','next']].max().max()
    plot([0, max_value], [0, max_value], alpha=0.1);
    xlim([0, max_value+0.1])
    ylim([0, max_value+0.1])

plot_time_map(df_diff);

def unique_chords(df):
    return sorted(df['chord'].unique())

for chord in unique_chords(df):
    print(chord)

import glob

files = glob.glob(data_dir + '*/*.lab')
tracks = pd.DataFrame({
    'album': [f.split('/')[-2].replace('_', ' ') for f in files],
    'name': [f.split('/')[-1].replace('.lab', '').replace('_', ' ') for f in files],
    'album_index': [int(f.split('/')[-2][:2]) for f in files]#,
#     'song_index': [int(f.split('/')[-1][:2]) for f in files]
})
tracks

def song_title(track):
    return ' / '.join(track[['album', 'name']])

def time_map_for_file(index):
    plot_time_map(add_differences(read_chord_file(files[index])).dropna())
    title(song_title(tracks.ix[index]))
    
time_map_for_file(5)

def add_track_id(df, track_id):
    df['track_id'] = track_id
    return df

selected_files = files
track_dfs = (read_chord_file(file) for file in selected_files)
track_dfs = (add_track_id(df, track_id) for (track_id, df) in enumerate(track_dfs))
track_dfs = (add_differences(df) for df in track_dfs)
all_events = pd.concat(track_dfs)
df_diff_all = all_events.dropna()

df_diff_all.head()

print('song count:', len(selected_files))
print('total diff event count in all songs:', len(df_diff_all))

df_diff_all.describe()

def outlier_quantiles(df, cols=['next','prev'], tolerance=0.01):
    df_nonzero = df[cols][df[cols] > 0]
    quantiles = df_nonzero.quantile([tolerance, 1 - tolerance])
    return quantiles

outlier_limits = outlier_quantiles(df_diff_all)
outlier_limits

def remove_outliers(df, limits, cols=['next','prev']):
    outlier_idxs = df['next'] == np.nan # just an array of False of proper length
    for col in cols:
        q_min, q_max = limits[col]
        print(q_min, q_max)
        series = df[col]
        idxs = series < q_min
        print(col, 'min', sum(idxs))
        outlier_idxs |= idxs
        idxs = series > q_max
        outlier_idxs |= idxs
        print(col, 'max', sum(idxs))
    print('outlier count:', sum(outlier_idxs), 'precentage:', sum(outlier_idxs) / len(df) * 100, '%')
    return df[~outlier_idxs]

df_diff_all_cleaned = remove_outliers(df_diff_all, outlier_limits)

df_diff_all_cleaned.describe()

plot_time_map(df_diff_all_cleaned, coloring=df_diff_all_cleaned['track_id'])

def inverse_polar(time_to_prev, time_to_next):
    # x = time_to_prev
    # y = time_to_next
    # (x, y) -> (r, phi) (cartesian to polar)
    # (r, phi) -> (velocity, acceleration) (no transform, just different interpretation)
    r = np.sqrt(time_to_prev**2 + time_to_next**2)
    phi = np.angle(time_to_next + 1j * time_to_prev) / (2 * np.pi)
    return (1 / (r / np.sqrt(2)), (phi - 0.125) * 8)

x = np.linspace(0, 1, 100)
plot(x, 1 - x)
scatter(*inverse_polar(x, 1 - x))
xlabel('velocity (r)')
ylabel('acceleration (phi)')
axes().set_aspect('equal');

def plot_inverse_polar_time_map(df_diff, coloring=None):
    cmap = plt.cm.get_cmap('RdYlBu')
    velocity, acceleration = inverse_polar(df_diff['prev'], df_diff['next'])
    c = np.linspace(0, 1, len(df_diff)) if coloring is None else coloring
    scatter(velocity, acceleration,
            alpha=0.5,
            c=c,
            cmap=cmap,
            edgecolors='none')
    xlabel('velocity')
    ylabel('acceleration')
    title('Time map')
    axes().set_aspect('equal')
    
    max_velocity = velocity.max()
    plot([0, 0], [max_velocity, 0], alpha=0.2);
    xlim([0, max_velocity+0.1])
    ylim([-1, 1])

plot_inverse_polar_time_map(df_diff);

plot_inverse_polar_time_map(df_diff_all_cleaned);

def plot_tracks(df, col, track_order=None):
    track_id = df['track_id']
    y = track_id
    if track_order is not None:
        mapping = track_order.argsort()
        y = y.apply(lambda x: mapping[x])
    plot(df[col], y, 'g.', label=col, alpha=0.1)
    xlabel(col)
    ylabel('track')

plot_tracks(df_diff_all, 'start')

def select_time_range(df, start, end, col='start'):
    series = df[col]
    return df[(series >= start) & (series <= end)]

plot_tracks(select_time_range(df_diff_all, 0, 100), 'start')

plot_tracks(df_diff_all_cleaned, 'next')

sns.distplot(df_diff_all_cleaned['next']);

next_medians = df_diff_all.groupby('track_id')['next'].median()
next_medians.describe()

tracks['next_median'] = next_medians
tracks_by_next_median = next_medians.argsort()
tracks.ix[tracks_by_next_median]

plot_tracks(df_diff_all, 'start', tracks_by_next_median)

scatter(tracks.ix[tracks_by_next_median]['album_index'], next_medians)
xlabel('album index')
ylabel('median of time-to-next within a song');

df = pd.DataFrame({
        'album': list(tracks['album_index'][df_diff_all_cleaned['track_id']]),
        'duration': list(df_diff_all_cleaned['next'])})
sns.violinplot(data=df, x='album', y='duration')
title('Distribution of chord segment durations (time-to-next) for each album');

total_lengths = df_diff_all.groupby('track_id').last()['end']

# indexes of last songs in each album
last_song_indexes = list(tracks[tracks['album_index'].diff() != 0].index)

scatter(np.arange(len(total_lengths)), total_lengths, c=tracks['album_index'], cmap=plt.cm.get_cmap('RdYlBu'))
for i in last_song_indexes:
    axvline(i, alpha=0.1)
title('Total length of songs')
xlabel('track id')
ylabel('length (sec)');

scatter(tracks['album_index'], total_lengths, c=tracks['album_index'], cmap=plt.cm.get_cmap('RdYlBu'))
title('Total length of songs')
xlabel('album index')
ylabel('length (sec)');

plot(sorted(total_lengths));
title('Songs ordered by total length')
xlabel('track id (reordered)')
ylabel('length (sec)');

total_lengths.describe()

print('shortest song:', total_lengths.min(), 'sec,', song_titles[total_lengths.argmin()])
print('longest song:', total_lengths.max(), 'sec,', song_titles[total_lengths.argmax()])

sns.distplot(total_lengths, bins=20)
axvline(total_lengths.median(), alpha=0.2)
xlabel('total length(sec)');

album_lengths = tracks.join(total_lengths).groupby('album_index').sum()['end']
album_lengths

stem(album_lengths)
title('Total album lengths');

chords = df_diff_all['chord'].value_counts()
print('unique chord count:', len(chords))
print('top 20 chords:')
chords[:20]

plot(chords)
title('chord frequency');



