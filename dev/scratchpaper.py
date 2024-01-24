import sys
sys.path.append('./')

from src.objects.workpiece import Workpiece

# Square, 10 cm wide beam 
path = list()
path.append([[0,0], [1,0]])
path.append([[1,0], [1,1]])
path.append([[1,1], [0,1]])
path.append([[0,1], [0,0]]) #Optional to close path
wkp_square = Workpiece(path=path)



def findCircleIntersections():
    pass

def calcAverageDepth():
    pass

def restructurePath():
    pass

