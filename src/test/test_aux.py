from src.objects.workpiece import Workpiece

def makeSquareWkp(width = 1.) -> Workpiece:
    """ Returns Square, 10 cm wide beam. """
    path = list()
    path.append([[0,0], [width,0]])
    path.append([[width,0], [width,width]])
    path.append([[width,width], [0,width]])
    path.append([[0,width], [0,0]]) #Optional to close path
    return Workpiece(path=path)

def makeCircularWkp(dia=1.) -> Workpiece:
    """ Returns circular, 10 cm diameter beam."""
    path = [
        [[-dia, 0.], [dia, 0.], [0., -dia]],
        [[dia, 0.], [-dia, 0.], [0., dia]]
    ]
    return Workpiece(path=path)