# get matplotlib configuration
get_ipython().magic('run plot_conf.py')

from data.VideoFolder import VideoFolder

my_data = VideoFolder('data/256min_data_set/')

def process(data):
    
    nb_videos = len(data.videos)
    frames_per_video = tuple(last - first + 1 for ((last, first), _) in data.videos)
    sorted_frames = sorted(frames_per_video)
    plt.plot(sorted_frames)
    plt.ylabel('Number of frames')
    plt.xlabel('Sorted video index')
    plt.ylim(ymin=0)
    print('There are', len(frames_per_video))
    print('The 10 shortest videos have', *sorted_frames[:10], 'frames')
    print('The 10 longest videos have', *sorted_frames[-10:], 'frames')
    
    return nb_videos, frames_per_video

(nb_videos, frames_per_video) = process(my_data)

def fit_distribution(frames_count, nb_videos, enough=1e3):
    plt.figure(1)
    n, bins, patches = plt.hist(frames_count, bins=50)
    bin_width = bins[1] - bins[0]
    plt.xlabel('Frame count')
    plt.ylabel('Video count')

    from scipy.stats import t
    import numpy as np
    param = t.fit(frames_count)
    x = np.linspace(min(frames_count),max(frames_count),500)
    area = len(frames_count) * bin_width
    pdf_fitted = t.pdf(x, param[0], loc=param[1], scale=param[2]) * area
    plt.plot(x, pdf_fitted)

    from scipy.stats import norm
    normal_pdf = norm.pdf(x, loc=param[-2], scale=param[-1]) * area
    plt.plot(x, normal_pdf, c='.6', linewidth=.5)
    p = norm.fit(frames_count)
    normal_pdf = norm.pdf(x, loc=p[-2], scale=p[-1]) * area
    plt.plot(x, normal_pdf, c='.3', linewidth=.5)

    plt.legend(('t-student', 'norm1', 'norm2', 'hist'))
    plt.title('Frame count distribution')


    # draw limits
    plt.figure(2)
    plt.axhline(y=0)
    plt.axhline(y=nb_videos)

    y = n.cumsum()

    plt.step(bins[1:], y)
    plt.title('Frame count cumulative distribution')
    plt.xlabel('Frame count *fc*')
    plt.ylabel('Nb of video with at least *fc* frames');
    
    print('nu: {:.2f}'.format(param[0]))
    print('Average length (frames): {:.0f}'.format(param[1]))
    print('90% interval: [{:.0f}, {:.0f}]'.format(*t.interval(0.90, param[0], loc=param[1], scale=param[2])))
    print('95% interval: [{:.0f}, {:.0f}]'.format(*t.interval(0.95, param[0], loc=param[1], scale=param[2])))
    print('\n')
    
    for i, p in enumerate(zip(n, bins)):
        print('{:2d} {:3.0f} {:3.0f}'.format(i, *p))
        if i >= enough - 1: break
    
fit_distribution(frames_per_video, nb_videos, enough=25)

# getting new stats
my_train_data = VideoFolder('data/processed-data/train/')

(nb_train_videos, frames_per_train_video) = process(my_train_data)

fit_distribution(frames_per_train_video, nb_train_videos, enough=25)

# get videos length and name
a = tuple((last - first + 1, i) for (i, ((last, first), _)) in enumerate(my_train_data.videos))
b = sorted(a)  # sort by length
print('5 longest videos', *b[-5:], sep='\n')
v = my_train_data.videos[b[-1][1]]
print('The longest video is:', v[1][0])
print('which has length: ', v[0][0] - v[0][1] + 1)

# getting new stats
my_val_data = VideoFolder('data/processed-data/val/')

(nb_val_videos, frames_per_val_video) = process(my_val_data)

my_sampled_data = VideoFolder('data/sampled-data/train/')

(nb_sampled_videos, frames_per_sampled_video) = process(my_sampled_data)

my_sampled_data.frames_per_class[:5]

[sum(my_sampled_data.frames_per_video[4*a:4*a+4]) for a in range(0,5)]

len(my_sampled_data.classes)

my_sampled_val_data = VideoFolder('data/sampled-data/val/', init_shuffle='init')

(nb_sampled_val_videos, frames_per_sampled_val_video) = process(my_sampled_val_data)

len(my_sampled_val_data.videos)

len(my_sampled_data.videos)

my_sampled_val_data.videos[:10]

my_sampled_data.videos[:10]



