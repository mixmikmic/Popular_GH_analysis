# Import some matplolib shortcuts for Jupyter notebook
get_ipython().magic('matplotlib inline')
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

from pypot.creatures import PoppyHumanoid

poppy = PoppyHumanoid()

from pypot.creatures import PoppyHumanoid

poppy = PoppyHumanoid(simulator='vrep')

poppy.stand_position.start()

poppy.motors

for m in poppy.motors:
    print(m.name)

poppy.l_elbow_y

poppy.l_elbow_y.present_temperature

[m.present_position for m in poppy.motors]

poppy.l_arm_z.goal_position = 50

poppy.l_arm_z.goal_position = -50

poppy.l_arm_z.moving_speed = 50

poppy.l_arm_z.goal_position = 90

poppy.l_arm_z.compliant = True

poppy.l_arm_z.compliant = False

[p.name for p in poppy.primitives]

poppy.upper_body_idle_motion.start()

import time

poppy.upper_body_idle_motion.start()
time.sleep(10)
poppy.upper_body_idle_motion.stop()



