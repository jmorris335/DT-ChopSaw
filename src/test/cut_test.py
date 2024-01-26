from src.objects.cut import Cut
from src.objects.chopsaw import ChopSaw
from src.objects.workpiece import Workpiece
from src.objects.blade import Blade
from src.objects.structure import Arm
from src.gui.plotter import *

import time

def cut_test():
    arm = Arm(h0=1.0, x_arm=0.0)
    blade = Blade(radius=0.50)
    # wkp = makeCircularWkp()
    wkp = makeSquareWkp()
    saw = ChopSaw(blade=blade, arm=arm)
    cut = Cut(saw=saw, wkp=wkp)
    cut.step()
    # saw.arm.x_arm += 0.1
    # cut.step()
    plotSawAndWkp(saw, wkp)

def makeSquareWkp() -> Workpiece:
    """ Returns Square, 10 cm wide beam. """
    path = list()
    path.append([[0,0], [1,0]])
    path.append([[1,0], [1,1]])
    path.append([[1,1], [0,1]])
    path.append([[0,1], [0,0]]) #Optional to close path
    return Workpiece(path=path)

def makeCircularWkp() -> Workpiece:
    """ Returns circular, 10 cm diameter beam."""
    path = [
        [[-.1, 0.], [.1, 0.], [0., -.1]],
        [[.1, 0.], [-.1, 0.], [0., .1]]
    ]
    return Workpiece(path=path)
