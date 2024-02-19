import numpy as np

from src.objects.workpiece import Workpiece

def makeSquareWkp(width = 1.) -> Workpiece:
    """ Returns Square, 10 cm wide beam. """
    path = list()
    path.append([[0,0], [width,0]])
    path.append([[width,0], [width,width]])
    path.append([[width,width], [0,width]])
    path.append([[0,width], [0,0]]) #Optional to close path
    return Workpiece(loops=path)

def makeNegativeSquareWkp(width = 1.):
    """Returns square wkp with CW path sequence."""
    path = [
        [[0, 0], [0, width]],
        [[0, width], [width, width]],
        [[width, width], [width, 0]],
    ]
    return Workpiece(loops = path)

def makeCircularWkp(dia=1.) -> Workpiece:
    """ Returns circular, 10 cm diameter beam."""
    path = [
        [[-dia, 0.], [dia, 0.], [0., -dia]],
        [[dia, 0.], [-dia, 0.], [0., dia]]
    ]
    return Workpiece(loops=path)

def makeQuarterCircle(radius=1.) -> Workpiece:
    """Returns a quadrant I quarter circle."""
    path = [
        [[0, 0], [radius, 0]],
        [[radius, 0], [0, radius], [radius*np.cos(np.pi/4), radius*np.sin(np.pi/4)]],
        [[0, radius], [0, 0]]
    ]
    return Workpiece(loops=path)

def makeGFSstrut(x=1.) -> Workpiece:
    loop = [
        [[0, 0], [2/5*x, 0]],
        [[2/5*x, 0], [2/5*x, 1/5*x]],
        [[2/5*x, 1/5*x], [3/10*x, 1/5*x]],
        [[3/10*x, 1/5*x], [1/2*x, 2/5*x]],

        [[1/2*x, 2/5*x], [7/10*x, 1/5*x]],
        [[7/10*x, 1/5*x], [3/5*x, 1/5*x]],
        [[3/5*x, 1/5*x], [3/5*x, 0]],
        [[3/5*x, 0], [x, 0]],
        [[x, 0], [x, 2/5*x]],
        [[x, 2/5*x], [4/5*x, 2/5*x]],
        [[4/5*x, 2/5*x], [4/5*x, 3/10*x]],
        [[4/5*x, 3/10*x], [3/5*x, 1/2*x]],

        [[3/5*x, 1/2*x], [4/5*x, 7/10*x]],
        [[4/5*x, 7/10*x], [4/5*x, 3/5*x]],
        [[4/5*x, 3/5*x], [x, 3/5*x]],
        [[x, 3/5*x], [x, x]],
        [[x, x], [3/5*x, x]],
        [[3/5*x, x], [3/5*x, 4/5*x]],
        [[3/5*x, 4/5*x], [7/10*x, 4/5*x]],
        [[7/10*x, 4/5*x], [1/2*x, 3/5*x]],

        [[1/2*x, 3/5*x], [3/10*x, 4/5*x]],
        [[3/10*x, 4/5*x], [2/5*x, 4/5*x]],
        [[2/5*x, 4/5*x], [2/5*x, x]],
        [[2/5*x, x], [0, x]],
        [[0, x], [0, 3/5*x]],
        [[0, 3/5*x], [1/5*x, 3/5*x]],
        [[1/5*x, 3/5*x], [1/5*x, 7/10*x]],
        [[1/5*x, 7/10*x], [2/5*x, 1/2*x]],

        [[2/5*x, 1/2*x], [1/5*x, 3/10*x]],
        [[1/5*x, 3/10*x], [1/5*x, 2/5*x]],
        [[1/5*x, 2/5*x], [0, 2/5*x]],
        [[0, 2/5*x], [0, 0]]
    ]
    return Workpiece(loops=loop)

def makeTwoWkp(x: float=1.):
    path = [
        [[0, 0], [4*x, 0]],
        [[4*x, 0], [4*x, x]],
        [[4*x, x], [x, x]],
        [[x, x], [x, 2*x]],
        [[x, 2*x], [4*x, 2*x]],
        [[4*x, 2*x], [4*x, 5*x]],
        [[4*x, 5*x], [0, 5*x]],
        [[0, 5*x], [0, 4*x]],
        [[0, 4*x], [3*x, 4*x]],
        [[3*x, 4*x], [3*x, 3*x]],
        [[3*x, 3*x], [0, 3*x]],
        [[0, 3*x], [0, 0]],
    ]
    return Workpiece(loops=path)