get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import pandas as pd
import sys
from collections import OrderedDict

sys.path.append('../../tools/music-processing-experiments')

from time_intervals import block_labels
from analysis import split_block_times

# let's generate some segments represented by their start time
def segments_from_events(event_times, labels=None):
    """Create a DataFrame with adjacent segments from a list of border events."""
    segments = pd.DataFrame({'start': event_times[:-1], 'end': event_times[1:]})
    segments['duration'] = segments['end'] - segments['start']
    segments = segments[['start', 'end', 'duration']]
    if labels != None:
        assert(len(event_times) == len(labels) + 1)
        segments['label'] = labels
    return segments

def events_from_segments(segments):
    """Create a list of border events DataFrame with adjacent segments."""
    return np.hstack([segments['start'], segments['end'].iloc[-1]])

def generate_segments(size, seed=0):
    np.random.seed(seed)
    event_times = np.random.normal(loc=1, scale=0.25, size=size+1).cumsum()
    event_times = event_times - event_times[0]
    return segments_from_events(event_times, np.random.randint(4, size=size))

segments = generate_segments(20)
segments.head()

events_from_segments(segments)

def plot_segments(segments, seed=42):
    size = len(segments)
    np.random.seed(seed)
    if 'label' not in segments.columns:
        colors = np.random.permutation(size) / size
    else:
        labels = segments['label']
        unique_labels = sorted(set(segments['label']))
        color_by_label = dict([(l, i) for (i, l) in enumerate(unique_labels)])
        norm_factor = 1.0 / len(unique_labels)
        colors = labels.apply(lambda l: color_by_label[l] * norm_factor)
    plt.figure(figsize=(20,5))
    plt.bar(segments['start'], np.ones(size), width=segments['duration'], color=cm.jet(colors), alpha=0.5)
    plt.xlim(0, segments['end'].iloc[-1])
    plt.xlabel('time')
    plt.yticks([]);

plot_segments(segments)

def make_blocks(total_duration, block_duration):
    return segments_from_events(np.arange(0, total_duration, block_duration))

total_duration = segments.iloc[-1]['end']
print('total duration:', total_duration)
blocks = make_blocks(total_duration, 0.25)
print('number of blocks:', len(blocks))
blocks.head()

plot_segments(blocks)

class Events():
    def __init__(self, start_times, labels):
        """last item must be sentinel with no label"""
        assert(len(labels) >= len(start_times) - 1)
        if len(labels) < len(start_times):
            labels = labels.append(pd.Series([np.nan]))
        self._df = pd.DataFrame({'start': start_times, 'label': labels}, columns=['start', 'label'])
    def df(self):
        return self._df
    
class Segments():
    def __init__(self, start_times, labels):
        """last item must be sentinel with NaN label"""
        self._df = segments_from_events(start_times, labels)

    def df(self):
        return self._df
    
    def join(self, other):
        sentinel_value = '_END_'
        def add_sentinel(df):
            last_event = df[-1:]
            return df.append(pd.DataFrame({
                'start': last_event['end'],
                'end': last_event['end'],
                'duration': 0.0,
                'label': sentinel_value
            }, columns=last_event.columns))
        def remove_sentinel(df, cols):
            for col in cols:
                df[col] = df[col].apply(lambda v: np.nan if v == sentinel_value else v)
        self_df = add_sentinel(self.df())[['start', 'label']].set_index('start')
        other_df = add_sentinel(other.df())[['start', 'label']].set_index('start')
        joined_df = self_df.join(other_df, lsuffix='_left', rsuffix='_right', how='outer')
        joined_df.fillna(method='ffill', inplace=True)
        remove_sentinel(joined_df, ['label_right', 'label_left'])
        joined_df['label_equals'] = joined_df['label_left'] == joined_df['label_right']
        joined_df.reset_index(inplace=True)
        joined_df['end'] = joined_df['start'].shift(-1)
        joined_df['duration'] = joined_df['end'] - joined_df['start']
        joined_df = joined_df[:-1]
        return joined_df #Segments(joined_df['start'], joined_df['label'])

annotations = Segments(np.array([0, 1, 2, 3, 3.5, 4]), ['A','B','A','C','A'])
annotations.df()

plot_segments(annotations.df())

estimations = Segments(np.array([0, 0.9, 1.8, 2.5, 3.1, 3.4, 4.5]), ['A','B','A','B','C','A'])
estimations.df()

plot_segments(estimations.df())

def join_segments(df1, df2):
    """Joins two dataframes with segments into a single one (ignoring labels)"""
    np.hstack(events_from_segments(df1), events_from_segments(df1))

events = np.hstack([events_from_segments(annotations), events_from_segments(estimations)])
events.sort()
events = np.unique(events)
events

merged = segments_from_events(events)
merged

plot_segments(merged)

merged_df = annotations.join(estimations)
merged_df

def chord_symbol_recall(pred_segments, true_segments):
    merged_df = pred_segments.join(true_segments)
    return merged_df[merged_df['label_equals']]['duration'].sum() / merged_df['duration'].sum()

chord_symbol_recall(estimations, annotations)



