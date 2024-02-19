from src.objects.cut import Cut
from src.objects.saw import Saw
from src.objects.blade import Blade
from src.objects.structure import Arm
from src.gui.plotter import *
from test.test_aux import *

def ct_main():
    cut_test()

def cut_test():
    # arm = Arm(h0_arm=0.1603881341161963, x_arm=0.005)
    # blade = Blade(radius_blade = 0.092)
    arm = Arm(h0_arm=1, x_arm=0, gap_arm=.125)
    blade = Blade(radius_blade=.25)
    # wkp = makeQuarterCircle(.1)
    # wkp = makeCircularWkp()
    wkp = makeSquareWkp(1)
    # wkp = makeNegativeSquareWkp(1)
    # wkp = makeGFSstrut(.25)
    # wkp = makeTwoWkp(1)
    saw = Saw(blade=blade, arm=arm)
    cut = Cut(saw=saw, wkp=wkp)
    cut.step()
    # cut.set(x_arm=0.001, h0_arm=0.5001)
    cut.step()
    # cut.step()
    plotStatic([cut])

def multipleLoops():
    path = [
        [[0, 0], [0, 2]],
        [[0, 2], [10, 2]],
        [[10, 2], [10, 0]]
    ]
    # path = [
    #     [[0, 0], [10, 0]],
    #     [[10, 0], [10, 2]],
    #     [[10, 2], [0, 2]]
        # [[0, 5], [10, 5]],
        # [[10, 5], [10, 7]],
        # [[10, 7], [0, 7]],

        # [[0, 0], [4, 0]],
        # [[4, 0], [4, -1]],
        # [[4, -1], [6, -1]],
        # [[6, -1], [6, 0]],
        # [[6, 0], [10, 0]],
        # [[10, 0], [10, 2]],
        # [[10, 2], [6, 2]],
        # [[6, 2], [6, 3]],
        # [[6, 3], [4, 3]],
        # [[4, 3], [4, 2]],
        # [[4, 2], [0, 2]],
        # [[0, 2], [0, 0]]
    # ]
    wkp = Workpiece(loops=path)
    arm = Arm(h0_arm=1, x_arm=5)
    blade = Blade(radius_blade = 2)
    saw = Saw(blade=blade, arm=arm)
    cut = Cut(saw=saw, wkp=wkp)
    cut.step()
    plotStatic([cut])
