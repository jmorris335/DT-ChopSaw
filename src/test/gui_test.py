import numpy as np

from src.gui.animator import *
from src.objects.saw import Saw
from src.objects.structure import Arm
from src.objects.blade import Blade
from src.objects.cut import Cut
from src.test.test_aux import *

def animationTest():
    wkp = makeSquareWkp(.15)
    saw = Saw()
    saw.set(x_arm=0, theta_arm=np.pi/4)
    cut = Cut(saw, wkp)
    action_bounds1 = {'x_arm' : [0, 0.05], 'theta_arm' : [np.pi/2, 0]}
    actions = makeLinearPath(action_bounds1, 50)

    animate(cut, actions, rate=1/30)
