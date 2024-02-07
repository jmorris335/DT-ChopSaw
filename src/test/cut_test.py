from src.objects.cut import Cut
from src.objects.saw import Saw
from src.objects.blade import Blade
from src.objects.structure import Arm
from src.gui.plotter import plotSawAndWkp
from src.test.test_aux import *

def cut_test():
    arm = Arm(h0_arm=.5, x_arm=1.5)
    blade = Blade(radius_blade=1)
    # wkp = makeQuarterCircle(.1)
    # wkp = makeCircularWkp()
    wkp = makeSquareWkp()
    saw = Saw(blade=blade, arm=arm)
    cut = Cut(saw=saw, wkp=wkp)
    # saw.arm.x_arm += 0.4
    cut.step()
    cut.set(x_arm=.9)
    cut.step()
    plotSawAndWkp(saw, wkp)
