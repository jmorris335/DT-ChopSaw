import numpy as np

from src.gui.animator import *
from src.objects.saw import Saw
from src.objects.structure import Arm
from src.objects.blade import Blade
from src.objects.cut import Cut
from src.test.test_aux import *

def animation_test():
    wkp = makeSquareWkp(.15)
    saw = Saw()
    saw.set(x_arm=0, theta_arm=np.pi/2)
    cut = Cut(saw, wkp)
    action_bounds1 = {'x_arm' : [0.08, 0.15], 
                      'theta_arm' : [np.pi/2, 0],
                      'torque' : [.1, .1]}
    action_bounds2 = {'x_arm' : [0.12, 0], 'torque' : [.1, .1]}
    actions1 = makeLinearPath(action_bounds1, 100)
    actions2 = makeLinearPath(action_bounds2, 100)
    actions = actions1

    animate(cut, actions, rate=.01)

    #TODO: Figure out why this is crashing
    #TODO: Link dynamic models
    #TODO: Add cut location to wkp
