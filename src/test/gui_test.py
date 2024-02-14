import numpy as np

from src.gui.plotter import *
from src.objects.saw import Saw
from src.objects.structure import Arm
from src.objects.blade import Blade
from src.objects.cut import Cut
from src.test.test_aux import *

def animation_test():
    # wkp = makeSquareWkp(.15)
    # wkp = makeGFSstrut(.25)
    # wkp = makeTwoWkp(.05)
    wkp = makeNegativeSquareWkp(.15)
    saw = Saw()
    saw.set(x_arm=0.0, theta_arm=np.pi/2)
    cut = Cut(saw, wkp)
    action_bounds1 = {'x_arm' : [0.0, .15], 
                      'theta_arm' : [np.pi/2, 0],
                      'torque' : [.1, .1]}
    action_bounds2 = {'x_arm' : [0.15, .15],
                      'theta_arm' : [0, 0], 
                      'torque' : [.1, .1]}
    action_bounds3 = {'x_arm' : [0.3, 0], 
                      'theta_arm' : [np.pi/4, 0], 
                      'torque' : [.1, .1]}
    actions1 = makeLinearPath(action_bounds1, 30)
    actions2 = makeLinearPath(action_bounds2, 30)
    actions3 = makeLinearPath(action_bounds3, 30)
    actions = actions1 + actions2

    animate(cut, actions, rate=.05)