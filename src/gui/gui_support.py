import numpy as np
from matplotlib.path import Path

import src.aux.geometry as geo

def arc2Bezier(A, B, C):
    """Converts an arc (terminated at points A and B and having point C on the
    curve as well) to a cubic bezier curve represented by 4 control points.
    
    Source: https://pomax.github.io/bezierinfo/#circles_cubic"""
    center = geo.calcCircleCenter(A, B, C)
    R = geo.calcCircleRadius(A, center)
    theta1, theta2 = [geo.calcAngleToAxis(*a) for a in [A, B]]
    if theta2 < theta1: theta2 += np.pi * 2
    theta = abs(theta2 - theta1)
    k = 4 / 3 * np.tan(theta)
    cntrl_pts = np.array([[R, 0],
                          [R, k],
                          [R*np.cos(theta) + R*k*np.sin(theta), R*np.sin(theta) - R*k*np.cos(theta)],
                          [R*np.cos(theta), R*np.sin(theta)]])
    cntrl_pts = np.matmul(cntrl_pts, [[np.cos(theta1), -np.sin(theta1)],
                                      [np.sin(theta1), np.cos(theta1)]]) #Rotate to arc position
    cntrl_pts += [center]*4 #Translate to original origin
    return cntrl_pts

def arcCommands():
    """Returns the commands corresponding to the 4 bezier control points in an arc."""
    return [Path.CURVE4, Path.CURVE4, Path.CURVE4]