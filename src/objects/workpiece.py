"""
| File: workpiece.py 
| Info: Presents a model for a constant-profile workpiece that can be cut on a saw.
| Author: John Morris, jhmrrs@clemson.edu  
| Organization: Product Lifecycle Management Center at Clemson University, plmcenter@clemson.edu  
| Permission: Copyright (C) 2023, John Morris. All rights reserved. Should not be reproduced, edited, sourced, or utilized without written permission from the author or organization
| Version History:
|- 0.1, 18 Jan 2024: Initialized
"""

import numpy as np

from src.aux.support import findDefault

class Workpiece:
    '''
    A workpiece that operations can be performed on. For now assumed to have a constant profile.

    Parameters
    ----------
    **kwargs : dict, optional
        Optional editing of state variables during initialization. Possible arguments are:

        E : float, default=69e9 Pa
            Young's modulus for the workpiece.
        path : list, default=list(); meters
            A list (len n) of lists (len 2,3) of lists (len 2). Each minor list represents a vertex
            (x,y coordinate pair) on the cross-section of the workpiece. The major list represents 
            each segment that makes up that cross-section profile. See note.

    Notes
    -----
    1. Creating the path: The cross-section is assumed to be constant throughout the piece. A 
    representative cross-section can be composed of line segments and arc segments. The first 
    two points are always the end points of a segment. Because segments must be in order,
    this means that the first point of any segment will always be the same as the second point in
    the segment preceeding it. Arc segments are given a third point, which can be any arbitrary 
    point on the arc. Note that full circles are not allowed, instead break the path into two 
    hemispheres.

    Example
    -------
        ```
        # Square, 10 cm wide beam 
        path = list()
        path.append([[0,0], [1,0]])
        path.append([[1,0], [1,1]])
        path.append([[1,1], [0,1]])
        path.append([[0,1], [0,0]]) #Optional to close path
        wkp_square = Workpiece(path=path)

        # Circular, 10 cm wide beam
        path = list()
        path.append([[1,0], [-1,0], [0,1]]) #Top hemisphere
        path.append([[-1,0], [1,0], [0,-1]]) #Bottom hemisphere
        wkp_circle = Workpiece(path=path)
        ```
    '''
    def __init__(self, **kwargs):
        self.E = findDefault(69e9, "E", kwargs)
        self.path = self.cleanPath(findDefault(list(), "path", kwargs))

    def cleanPath(self, path: list=None) -> list:
        """
        Cleans the list of points that comprise the cross-sectional path. Will not perfectly 
        make a profile, but will close open profiles, and check that all points are valid coordinate
        tuples. 

        Parameters
        -----------
        points : list, optional
            A list of tuples all with a length of 2, representing the vertices (x, y) of
            the profile of the workpiece. If not provided, function accesses internal path member.

            Note origin of the coordinate system is at the intersection of the table and guard of the saw.
            Any coordinate set will be shifted so that the leftmost x and bottommost y is 0.
        """
        if path is None: path = self.path
        if len(path) < 2: raise(Exception("Not enough segments provided in path"))
        if tuple(path[-1][1]) != tuple(path[0][0]): 
            path.append((path[-1][2], path[0][0]))
        path = self.shiftPositive(path)
        return path
    
    def shiftPositive(self, path: None) -> list:
        """Shifts the points so that all points are in the First Quadrant (positive) and, 
        if possible, at least 1 point is at the origin. Only shifts in the XY plane."""
        if path is None: path = self.path
        minx = self.findMinimumCoord(path, index=0)
        miny = self.findMinimumCoord(path, index=1)
        for seg in path:
            for point in seg:
                point[0] += minx
                point[1] += miny
        return path

    def findMinimumCoord(self, path: list=None, index: int=0):
        """Returns the element with the most negative coordinate in the points set.

        Parameters
        ----------
        points : list, optional
            A list of points, can be a list of tuples.
        index : int, default=0
            The index of the coordinate to minimize.

        Return
        ------
        any: the minimum coordinate at the specified index in the set
        """
        if path is None: path = self.path
        min_pt = path[0][0][0]
        for seg in path:
            if len(seg) == 2: #line
                a = self.findMinimumCoordOnLine(seg, index) 
            else: #arc
                a = self.findMinimalPointOnArc(seg, index)
            if a < min_pt: 
                min_pt = a
        return min_pt
    
    def findMinimumCoordOnLine(self, seg, index: int=0):
        """Returns the most negative element in the segment."""
        coord_set = [p[index] for p in seg if hasattr(p, "__len__") and len(p) > index]
        return min(coord_set)

    def findMinimalPointOnArc(self, seg, index: int=0):
        """Returns the most negative point on an arc, which is either one of the endpoints or
        the minimum point on the great circle."""
        if len(seg) != 3: return self.findMinimumCoord(seg, index)
        A, B, D = seg
        C = self.calcCircleCenter(A, B, C)
        points = np.array([A, B, D]) - C
        theta_set = [self.calcAngleToMin(a[0], a[1], index) for a in points]
        theta_D = theta_set[-1]
        if min(theta_set) == theta_D or max(theta_set) == theta_D:
            R = self.calcCircleRadius(A, C)
            return C[index] - R
        else:
            return min(A[index], B[index])
        
    @staticmethod
    def calcCircleRadius(A, C) -> float:
        """Calculates the radius of a circle where A is a point on the circle and C is the center
        of the circle."""
        return np.sqrt((A[0] - C[0])**2 + (A[1] - C[1])**2)

    @staticmethod
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

    @staticmethod
    def calcAngleToMin(x, y, index: int=0) -> float:
        """Calculates the angle (in radians) between the vector formed by the origin and the 
        point located at (x, y) and either the negative x-axis (index=0) or negative y-axis (index=1)"""
        if index != 0: index = 1
        if x == 0: theta = 0 if y == -1 else np.pi
        else: theta = np.arctan(y/x)
        if index: theta -= np.pi/2
        if theta < 0: theta += 2*np.pi
        return theta
        
if __name__ == '__main__':
    # Square, 10 cm wide beam 
    path = list()
    path.append([[0,0], [1,0]])
    path.append([[1,0], [1,1]])
    path.append([[1,1], [0,1]])
    path.append([[0,1], [0,0]]) #Optional to close path
    wkp_square = Workpiece(path=path)
