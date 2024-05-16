# Import some matplolib shortcuts for Jupyter notebook
get_ipython().magic('matplotlib inline')
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

from pypot.creatures import PoppyHumanoid

poppy = PoppyHumanoid()

poppy.init_position.start()

from pypot.primitive.move import MoveRecorder

recorder = MoveRecorder(poppy, 50, poppy.l_arm)

poppy.alias

for m in poppy.l_arm:
    m.compliant = True

recorder.start()

recorder.stop()

recorder.move

print('{} key frames have been recorded.'.format(len(recorder.move.positions())))

from copy import deepcopy

my_first_move = deepcopy(recorder.move)

ax = axes()
my_first_move.plot(ax)

recorder = MoveRecorder(poppy, 50, poppy.motors)

for m in poppy.motors:
    m.compliant = True

import time

recorder.start()
time.sleep(10)
recorder.stop()

my_second_move = deepcopy(recorder.move)

poppy.init_position.start()

from pypot.primitive.move import MovePlayer

player = MovePlayer(poppy, my_first_move)

player.start()

player = MovePlayer(poppy, my_second_move)
player.start()

for move in [my_first_move, my_second_move]:
    player = MovePlayer(poppy, move)
    
    player.start()
    player.wait_to_stop()

with open('my-first-demo.move', 'w') as f:
    my_first_move.save(f)

get_ipython().system('head -n 20 my-first-demo.move')

from pypot.primitive.move import Move

with open('my-first-demo.move') as f:
    my_loaded_move = Move.load(f)

from IPython.display import YouTubeVideo

YouTubeVideo('https://youtu.be/Hy56H2AZ_XI?list=PLdX8RO6QsgB6YCzezJHoYuRToFOhYk3Sf')

from IPython.display import YouTubeVideo

YouTubeVideo('hEBdz97FhS8')



