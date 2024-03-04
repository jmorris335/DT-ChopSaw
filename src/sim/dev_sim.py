from src.objects.cut import Cut
from src.objects.saw import Saw
from src.objects.blade import Blade
from src.objects.structure import Arm
from src.gui.plotter import *
from test.test_aux import *
from src.db.logger import Logger

def dev_sim_main():
    resetDB()
    animation_test()

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

def animation_test():
    # wkp = makeSquareWkp(.1)
    wkp = makeGFSstrut(.1)
    # wkp = makeTwoWkp(.05)
    # wkp = makeNegativeSquareWkp(.15)
    saw = Saw()
    saw.set(x_arm=0.0, theta_arm=np.pi/2)
    cut = Cut(saw, wkp)
    action_bounds1 = {'x_arm' : [0.0, .15], 
                      'theta_arm' : [np.pi/2, 0],
                      'torque' : [-.1, -.1]}
    action_bounds2 = {'x_arm' : [0.15, 0],
                      'theta_arm' : [0, 0], 
                      'torque' : [-.1, -.1]}
    action_bounds3 = {'x_arm' : [0.3, 0], 
                      'theta_arm' : [np.pi/4, 0], 
                      'torque' : [-.1, -.1]}
    actions1 = makeLinearPath(action_bounds1, 50)
    actions2 = makeLinearPath(action_bounds2, 50)
    actions3 = makeLinearPath(action_bounds3, 30)
    actions = actions1 + actions2

    # fig, axs = plt.subplots(nrows=1, ncols=2)
    # animate(cut, actions, rate=.01, block=False, fig=fig, ax=axs[0])
    animateWithData(cut, actions, rate=0.05, logger=cut.logger, y_names=[ "cut_depth", "load"])

def resetDB():
    l = Logger(Blade())
    l.resetDB()