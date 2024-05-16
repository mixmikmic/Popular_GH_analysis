from IPython.display import YouTubeVideo
YouTubeVideo('bxe2T-V8XRs')

get_ipython().magic('pylab inline')

# X = (hours sleeping, hours studying), y = Score on test
X = np.array(([3,5], [5,1], [10,2]), dtype=float)
y = np.array(([75], [82], [93]), dtype=float)

X

y

X = X/np.amax(X, axis=0)
y = y/100 #Max test score is 100

X

y

from IPython.display import Image
i = Image(filename='images/simpleNetwork.png')
i

