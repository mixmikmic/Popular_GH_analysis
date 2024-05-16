import sys

'This is %d %s bird!' % (1, 'dead')

'%(number)d more %(food)s' % {'number' : 1, 'food' : 'burger'}

'My {1[kind]} runs {0.platform}'.format(sys, {'kind': 'laptop'})

