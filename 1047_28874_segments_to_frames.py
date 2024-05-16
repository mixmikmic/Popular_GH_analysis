get_ipython().magic('pylab inline')
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
pylab.rcParams['figure.figsize'] = (12, 6)

segments = pd.DataFrame.from_records([
    (0, 2.4, 'C'),
    (2.4, 2.8, 'F'),
    (2.8, 4.7, 'G'),
    (4.7, 5.2, 'C'),
], columns=['start','end','label'])
segments

segment_count = len(segments)
total_duration = segments['end'].iloc[-1]
frame_duration = 1.0
hop_duration = 0.5

def time_intervals(segments):
    return [(v['start'], v['end']) for (k,v) in segments[['start', 'end']].iterrows()]

def plot_segments(time_intervals):
    ax = plt.gca()
    for (i, (s, e)) in enumerate(time_intervals):
        j = (i / 5) % 1
        yshift = 0.1 * (abs(j - 0.5) - 0.5)
        ax.add_patch(Rectangle(
                (s, yshift), e-s, yshift + 1, alpha=0.5, linewidth=2,
                edgecolor=(1,1,1), facecolor=plt.cm.jet(j)))
    pad = 0.1
    xlim(0 - pad, total_duration + pad)
    ylim(0 - pad, 1 + pad)

plot_segments(time_intervals(segments))

def frame_count(total_duration, frame_duration, hop_duration):
    return math.ceil((max(total_duration, frame_duration) - frame_duration) / hop_duration + 1)

frame_count(total_duration, frame_duration, hop_duration)

def frames(total_duration, frame_duration, hop_duration):
    count = frame_count(total_duration, frame_duration, hop_duration)
    return [(i * hop_duration, i * hop_duration + frame_duration) for i in range(count)]

def frame_centers(total_duration, frame_duration, hop_duration):
    count = frame_count(total_duration, frame_duration, hop_duration)
    return [(0.5  * frame_duration+ i * hop_duration) for i in range(count)]

f_centers = frame_centers(total_duration, frame_duration, hop_duration)
f_centers

f = frames(total_duration, frame_duration, hop_duration)
f

plot_segments(f)

def label_at_time(time, segments):
    labels = segments[(segments['start'] <= time) & (segments['end'] >= time)]['label']
    if len(labels) >= 0:
        return labels.iloc[0]

[label_at_time(t, segments) for t in f_centers]



