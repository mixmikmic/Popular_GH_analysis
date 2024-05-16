

# import matplotlib, numpy
import matplotlib.pyplot as plt
import numpy as np

# set plt.figure()
fig = plt.figure(figsize=(10, 8))

ax = fig.add_subplot(111)
fig.subplots_adjust(top=0.85)

ax.set_title('axes title')

ax.set_ylabel('Frequency')
ax.set_xlabel('Data')

ax.text(4, 7, 'Some text', style='italic',
        bbox={'facecolor':'red', 'alpha':0.5, 'pad':10})

ax.plot(np.linspace(0, 1, 5), np.linspace(0, 5, 5))
ax.plot([4], [1], 's')
ax.annotate('Here is the point', xy=(4, 1), xytext=(3, 4),
            arrowprops=dict(facecolor='blue', shrink=0.04))

ax.axis([0, 10, 0, 10])

plt.show()

# more advance example
fig = plt.figure()
ax = fig.add_subplot(111)

t = np.arange(0.0, 5.0, 0.01)
s = np.cos(2*np.pi*t)
line, = ax.plot(t, s, lw=2)

ax.annotate('local max', xy=(2, 1), xytext=(3, 1.5),
            arrowprops=dict(facecolor='green', shrink=0.05),
            )

ax.set_ylim(-2,2)
plt.show()



