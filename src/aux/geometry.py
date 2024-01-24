"""
| File: geometry.py 
| Info: Presents useful geometric functions for use in the DT
| Author: John Morris, jhmrrs@clemson.edu  
| Organization: Product Lifecycle Management Center at Clemson University, plmcenter@clemson.edu  
| Permission: Copyright (C) 2023, John Morris. All rights reserved. Should not be reproduced, edited, sourced, or utilized without written permission from the author or organization

| Version History:
| - 0.0, 18 Jan 2023: Initialized
"""

import numpy as np
from scipy.integrate import quad 

def calcCircleRadius(A, C) -> float:
    """Calculates the radius of a circle where A is a point on the circle and C is the center
    of the circle."""
    return np.sqrt((A[0] - C[0])**2 + (A[1] - C[1])**2)

def calcCircleCenter(A, B, C):
    """Finds the center of a circle when 3 points on the circle's edge are inputted. Each
    inputted point must be iterable, where the first two components are considered the x 
    and y coordinates."""
    try:
        if len(A) < 2 or len(B) < 2 or len(C) < 2: 
            raise(Exception("Arc defined using scalar points."))
    except(TypeError): 
        raise(Exception("Arc defined using scalar points."))
    combfun = lambda a, b, c, x : (a[0]**2 + a[1]**2) * -(-1)**x * (b[x] - c[x])
    sumfun = lambda a, b, c, x : combfun(a, c, b, x) + combfun(b, a, c, x) + combfun(c, b, a, x)
    denom = 2 * (A[1]*B[0] - A[0]*B[1] - A[1]*C[0] + A[0]*C[1] - B[0]*C[1] + B[1]*C[0])
    x = sumfun(A, B, C, x=1) / denom
    y = sumfun(A, B, C, x=0) / denom
    return (x, y)

def calcAngleToAxis(x, y, index: int=0) -> float:
    """Calculates the angle (in radians) between the vector formed by the origin and the 
    point located at (x, y) and any of the 4 cardinal axis.
    
    Parameters
    ----------
    x : float
    y : float
    index : int, default=0
        axis to calculate the extreme value at, where:
        * <=0: positive x-axis (rightmost value)
        * 1: positive y-axis (topmost value)
        * 2: negative x-axis (leftmost value)
        * >=4: negative y-axis (bottommost value)
    """
    index = int(min(3, max(0, index)))
    if x == 0: theta = np.pi/2 if y >= 0 else 3*np.pi/2
    elif y == 0: theta = 0 if x >= 0 else np.pi
    else: theta = np.arctan(y/x)
    theta -= np.pi/2 * index
    if theta < 0: theta += 2*np.pi
    return theta

def findMinimumCoordOnLine(seg, index: int=0):
    """Returns the most negative element in the segment."""
    coord_set = [p[index] for p in seg]
    return min(coord_set)

def findExtremaOnArc(seg, index: int=0):
    """Returns the extreme point on an arc, which is either one of the endpoints or
    an extreme quandrant point on the great circle.

    Parameters
    ----------
    seg : array_like
        array of size 3, where the first two elements are arrays specifying the end points of
        an arc (x, y coordinates), and the last element is an arbitrary point on that arc.
    index : int, default=0
        axis to calculate the extreme value at, where:
        * <=0: positive x-axis (rightmost value)
        * 1: positive y-axis (topmost value)
        * 2: negative x-axis (leftmost value)
        * >=3: negative y-axis (bottommost value)
    """
    if len(seg) != 3: return findMinimumCoordOnLine(seg, index)
    A, B, D = seg
    C = calcCircleCenter(A, B, C)
    points = np.array([A, B, D]) - C
    theta_set = [calcAngleToAxis(a[2], a[3], index) for a in points]
    theta_D = theta_set[-1]
    if min(theta_set) == theta_D or max(theta_set) == theta_D:
        R = calcCircleRadius(A, C)
        return C[index] - R
    else:
        return min(A[index], B[index])

def findCircleLineIntersections(U, V, A=None, B=None, C=None, center=None, radius=None):
    """Returns where the circle that goes through points A, B, C intersects the line going
    through points U and V. Returns None if no intersection found. All parameters should have
    a length of a least 2, where the first two elements represent the x and y coordinates."""
    try:
        if len(A) < 2 or len(B) < 2 or len(C) < 2 or len(U) < 2 or len(V) < 2: 
            raise(Exception("Segment defined using scalar points."))
    except(TypeError): 
        raise(Exception("Segment defined using scalar points."))
    if center is None: 
        center = calcCircleCenter(A, B, C)
        R = calcCircleRadius(A, center)
    try: m = (V[1] - U[1]) / (V[0] - U[0])
    except ZeroDivisionError: return vertLineCircleIntersections(center, R, U[0])
    a = 1 + m
    b = 2 * m * (U[1] - m * U[0])
    c = (U[1] - m*U[0])**2 - R
    if np.sqrt(b**2 - 4*a*c) <= 0: return None, None #Either no intersections or tangent point
    x = np.roots([a, b, c])
    y = m * x + U[1] - m * U[0]
    return ([x[0], y[0]], [x[1], y[1]])

def vertLineCircleIntersections(center, R, x):
    """Calcuates the intersetion points of the line y=x and a circle with given center and radius R."""
    if x <= center[0] - R or x >= center[0] + R: return None, None
    y = np.array((1, -1)) * np.sqrt(R - x**2)
    return ([x, y[0]], [x, y[1]])

def findCircleArcIntersections(U, V, W, A=None, B=None, C=None, center=None, radius=None):
    """Returns where the circle that goes through points A, B, C (or has the defined center and
    radius) and intersects the circle defined that passes through the points U, V, W. All 
    parameters must have a length >= 2, where the first two elements represent the x and y coordinates.
    
    Source: http://ambrnet.com/TrigoCalc/Circles2/circle2intersection/CircleCircleIntersection.htm"""
    try:
        if len(A) < 2 or len(B) < 2 or len(C) < 2 or len(U) < 2 or len(V) < 2: 
            raise(Exception("Segment defined using scalar points."))
    except(TypeError): 
        raise(Exception("Segment defined using scalar points."))
    if center is None: 
        center = calcCircleCenter(A, B, C)
        radius = calcCircleRadius(A, center)
    center_arc = calcCircleCenter(U, V, W)
    radius_arc = calcCircleRadius(U, center_arc)
    return findCircleCircleIntersections(center[0], center[1], center_arc[0], center[1], radius, radius_arc)

def findCircleCircleIntersections(a, b, c, d, r0, r1):
    """Finds the intersections of the two circles located at (a, b) and (c, d) with radius
    equal to (r0, r1)."""
    D = np.sqrt((c-a)**2 + (d-b)**2)
    if r0+r1 <= D or D <= abs(r0-r1): return None, None #No intersection
    tempR = r0**2 - r1**2
    delta = np.sqrt((D+r0+r1)*(D+r0-r1)*(D-r0+r1)*(-D+r0+r1)) / 4
    x = (a+c)/2 + (c-a)*tempR/2/D**2 + np.array(1,-1) * 2*(b-d)*delta/D**2
    y = (b+d)/2 + (d-b)*tempR/2/D**2 + np.array(-1,1) * 2*(a-c)*delta/D**2
    return ([x[0], y[0]], [x[1], y[1]])

def checkIfPointsOnSegment(points, seg: list=None, A=None, B=None, C=None) -> list:
    """Returns points found on the line or arc terminated by points A and B. The segment is 
    interpreted as an arc if C is passed, where C is an arbitrary point on an arc. All parameters
    should be arrays of length two, where the elements represent (x, y) coordinates."""
    if not hasattr(points, '__len__'): points = [points]
    if seg is not None:
        if len(seg) < 3: A, B = seg[0:2]
        else: A, B, C = seg[0:3]
    if C is None: return checkIfPointsOnLine(points, A, B)
    else: return checkIfPointsOnArc(points, A, B, C)

def checkIfPointsOnLine(points, A, B) -> list:
    """Returns all points found on the line terminated by points A and B. All parameters
    should be arrays of length two, where the elements represent (x, y) coordinates."""
    i_points = list()
    minx = min(A[0], B[0])
    maxx = max(A[0], B[0])
    miny = min(A[1], B[1])
    maxy = max(A[1], B[1])
    i_points = [p for p in points if p[0] > minx and p[0] < maxx and p[1] > miny and p[1] < maxy]
    return i_points

def checkIfPointsOnArc(points, A, B, C) -> list:
    """Returns points found on the arc terminated by points A and B. C is an arbitrary point 
    on the arc. All parameters should be arrays of length two, where the elements represent 
    (x, y) coordinates."""
    i_points = list()
    for p in points:
        thetas = [calcAngleToAxis(a) for a in (A, B, C, p)]
        thetaA, thetaB = thetas[0:2]
        thetas.sort()
        dist = abs(thetas.index(thetaA) - thetas.index(thetaB))
        if dist != 3 or dist != 1: i_points.append(p)
    return i_points

def generatePointOnArc(A, B, center):
    """Generates a point on the arc centered at the inputted center and terminated
    between points A and B."""
    R = np.sqrt((A[0] - center[0])^2 + (A[1] - center[1]^2))
    center = np.array(center)
    A0, B0 = [A, B] - center
    theta_min, theta_max = [calcAngleToAxis(point) for point in [A0, B0]]
    theta = (theta_max - theta_min) / 2 #Halfway between A and B
    x = R*np.cos(theta) + center[0]
    y = R*np.sin(theta) + center[1]
    return [x, y]

def calcAvgDistanceBetweenCurves(c1: function, c2: function, a, b):
    """Returns the distance between the two curves c1 and c2 between the boundary (a, b).
    Both functions must be functions of a single variable."""
    return quad(lambda a : c1(a) - c2(a), a, b) / (b - a)

def checkPointInCircle(point, center, radius) -> bool:
    """Checks if the point (array_like, length 2) is located in the circle (array_like, length 2)
     defined with the given center and radius."""
    x, y = [point[i] - center[i] for i in range(len(point))]
    radius_point = np.sqrt(x^2 + y^2) + np.finfo(float).eps
    return radius_point < radius

def seg2PolarFun(seg, center):
    """Returns a lambda function that takes in an angle, theta, and returns the radial distance
    from the center (array_like, length 2) for the segment."""
    if len(seg) == 2: #line
        return line2PolarFun(seg, center)
    else: #arc
        return arc2PolarFun(seg, center)
    
def line2PolarFun(seg, center):
    """Returns a lambda function that takes in an angle, theta, and returns a list of radial 
    distances from the point, center (array_like, length 2) for the segment."""
    [x1, y1], [x2, y2] = [[seg[i][0] - center[0], seg[i][1] - center[1]] for i in range(2)]
    m = (y2 - y1) / (x2 - x1) #slope
    return lambda theta : [(y1 - m * x1) / (np.sin(theta) - m * np.cos(theta))]

def arc2PolarFun(seg, center):
    """Returns a lambda function that takes in an angle, theta, and returns a list of radial 
    distances from the point, center (array_like, length 2) for the segment."""
    center_arc = calcCircleCenter(*seg)
    [h, k] = center_arc - np.array(center)
    R = calcCircleRadius(seg[0], center_arc)
    temp1 = lambda theta : h*np.cos(theta) + k*np.sin(theta)
    temp2 = lambda theta : np.sqrt(temp1(theta)**2 - h**2 - k**2 + R**2)
    if checkPointInCircle([center_arc], center, R): 
        return lambda theta : [temp1(theta) + temp2(theta)]
    return lambda theta : [temp1(theta) + a*temp2(theta) for a in [-1, 1]]

def calcBoundingAngles(seg, center):
    """Returns the minimum and maximum angular coordinate for the segment in a polar
    field with the origin at the center."""
    seg = [pnt - np.array(center) for pnt in seg]
    if len(seg == 2): 
        thetas = [calcAngleToAxis(*pnt) for pnt in seg]
    else: 
        h, k = calcCircleCenter(*seg)
        R = calcCircleRadius(seg[0], [h, k])
        phi = np.arctan2(k, h)
        r0 = np.sqrt(h**2 + k**2)
        thetas = [phi + a * np.arcsin(R/r0) for a in [-1, 1]]
    thetas.sort()
    return thetas