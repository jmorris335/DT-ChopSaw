"""
| File: workpiece.py 
| Info: Presents a model for a constant-profile workpiece that can be cut on a saw.
| Author: John Morris, jhmrrs@clemson.edu  
| Organization: Product Lifecycle Management Center at Clemson University, plmcenter@clemson.edu  
| Permission: Copyright (C) 2023, John Morris. All rights reserved. Should not be reproduced, edited, sourced, or utilized without written permission from the author or organization
| Version History:
| - 0.1, 18 Jan 2024: Initialized
"""

import numpy as np
from matplotlib.patches import PathPatch
from matplotlib.path import Path

import src.aux.geometry as geo
import src.gui.gui_support as gui
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
        L : float, default=1 meter
            Length of the workpiece.
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

    TODO: Make the path member a function that returns the profile at a specific length
    TODO: Make a function that transforms the profile path for different miter angles
    '''
    def __init__(self, **kwargs):
        self.E = findDefault(69e9, "E", kwargs)
        self.L = findDefault(1., "L", kwargs)
        self.path = self.cleanPath(findDefault(defaultPath(), "path", kwargs))

        # GUI operations
        self.patches = self.plot()

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
    
    def set(self, **kwargs):
        """Determines if any passed keyword arguments are attributes of the entity, and 
        sets them if so."""
        for key, val in kwargs.items():
            attr = getattr(self, key, None)
            if attr is not None:
                setattr(self, key, val)

    def step(self, **kwargs):
        pass
    
    def shiftPositive(self, path: None) -> list:
        """Shifts the points so that all points are in the First Quadrant (positive) and, 
        if possible, at least 1 point is at the origin. Only shifts in the XY plane."""
        if path is None: path = self.path
        minx = self.findMinimumCoord(path, index=0)
        miny = self.findMinimumCoord(path, index=1)
        for seg in path:
            for point in seg:
                point[0] -= minx
                point[1] -= miny
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
                a = geo.findMinimumCoordOnLine(seg, index) 
            else: #arc
                a = geo.findExtremaOnArc(seg, index+2)
            if a < min_pt: 
                min_pt = a
        return min_pt
    
    def plot(self) -> PathPatch:
        """Returns a matplotlib PathPatch object defining the boundary of the object."""
        path = self.makePath()
        patch = PathPatch(path, fc="silver", lw=3, ec="black", label="Workpiece")
        return [patch]
    
    def makePath(self) -> Path:
        """Generates Path object from workpiece path."""
        if len(self.path) == 0: return Path([[0,0]], [Path.MOVETO])
        vertices = [self.path[0][0]]
        codes = [Path.MOVETO]
        for seg in self.path:
            if len(seg) == 2: 
                vertices.append(list(seg[1]))
                codes.append(Path.LINETO)
            else:
                vertices.extend(gui.arc2Bezier(*seg)[1:])
                codes.extend(gui.arcCommands())
        return Path(vertices, codes)
    
    def updatePatches(self):
        """Updates the patch of the wkp with wkp path."""
        path = self.makePath()
        self.patches[0].set_path(path)

    def pointInPath(self, P) -> bool:
        """Returns True if the point lies within the workpiece path. Points on the boundary are
        treated as being outside the path.
        
        Process
        -------
        1. Create bounding box around path.
        1. Identify point on a horizontal line the left of the bounding box (ensured to be outside
        the profile).
        1. Make a ray from the outside point to point P and count how many times the path intersects
        the ray. An odd number of intersections indicates that P is on the interior of the path. 

        Edge Cases
        ----------
        - Intersections with horizontal lines are treated as no intersection occuring. 
        - For intersections with vertices, only odd segments are counted.
        """
        x_outside = self.findMinimumCoord(index=0) - 1
        A = (x_outside, P[1])

        crosses = 0
        for seg in self.path:
            if len(seg) < 3:
                intx_pts = [geo.findLineLineIntersections(A, P, *seg)]
            else:
                intx_pts = geo.findCircleLineIntersections(A, P, *seg)

            for intx_pt in intx_pts:
                if intx_pt is None: continue
                if len(seg) < 3:
                    bounded = len(geo.checkIfPointsOnLine([intx_pt], *seg[:2])) > 0
                else:
                    bounded = len(geo.checkIfPointsOnArc([intx_pt], *seg))
                bounded *= len(geo.checkIfPointsOnLine([intx_pt], A, P)) > 0
                if not bounded: continue
                
                at_vertex = np.sqrt(sum([(intx_pt[i] - seg[0][i]) ** 2 for i in range(2)])) < geo.eps
                at_vertex += np.sqrt(sum([(intx_pt[i] - seg[1][i]) ** 2 for i in range(2)])) < geo.eps
                if at_vertex:
                    if self.path.index(seg) % 2: crosses += 1
                else:
                    crosses += 1
        
        return bool(crosses % 2)

def defaultPath():
    """Makes the path of a square, 10 cm wide beam."""
    path = [[[0,0], [1,0]]]
    path.append([[1,0], [1,1]])
    path.append([[1,1], [0,1]])
    path.append([[0,1], [0,0]])
    return path
