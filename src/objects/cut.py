"""
| File: cut.py 
| Info: Presents the state-model for a cut made with a chopsaw on a homogeneous workpiece 
| Author: John Morris, jhmrrs@clemson.edu  
| Organization: Product Lifecycle Management Center at Clemson University, plmcenter@clemson.edu  
| Permission: Copyright (C) 2023, John Morris. All rights reserved. Should not be reproduced, edited, sourced, or utilized without written permission from the author or organization
| Version History:
| - None
"""

import numpy as np

from src.aux.support import findDefault, loopSlice
import src.aux.geometry as geo
from src.objects.chopsaw import ChopSaw
from src.objects.workpiece import Workpiece

class Cut():
    '''
    Models the act of cutting a workpiece with a saw blade.

    Parameters
    ----------
    saw : ChopSaw, default=ChopSaw()
        The saw performing the cut, constructs a new ChopSaw object by default.
    wkp : Workpiece, default=Workpiece()
        The workpiece being cut, constructs a new Workpiece object by default.
    **kwargs : dict, optional
        Optional editing of state variables during initialization. Possible arguments are:

        h : float; meters
            Current workpiece height at place of cut.
        V : float; rad/s
            Cutting speed (tangential speed of the saw blade)
        t_chip : float; meters
            The undeformed chip thickness (average depth of the saw blade cut).

    References
    ----------
    1. Kapoor, S. G., R. E. DeVor, R. Zhu, R. Gajjela, G. Parakkal, and D. Smithey. “Development of Mechanistic Models for the Prediction of Machining Performance: Model Building Methodology.” Machining Science and Technology 2, no. 2 (December 1, 1998): 213-38. https://doi.org/10.1080/10940349808945669.
    '''
    def __init__(self, saw=ChopSaw(), wkp=Workpiece(), **kwargs):
        self.id = findDefault("0", "id", kwargs)

        # Components
        self.saw = saw
        self.wkp = wkp

        # Physical Values
        self.alpha_n = findDefault(self.saw.blade.rake, "alpha_n", kwargs)
        # self.t_c = findDefault(self.saw.blade.kerf, "t_c", kwargs)

        # Dyanmic Values
        # self.h = findDefault(self.H, "h", kwargs)

        # Inputs
        self.V = findDefault(self.saw.blade.omega*self.saw.blade.radius, "V", kwargs)
        self.x_arm = findDefault(self.saw.arm.x_arm, "x_arm", kwargs)
        self.theta_arm = findDefault(self.saw.arm.theta_arm, "theta_arm", kwargs)
        self.phi_arm = findDefault(self.saw.arm.phi_arm, "phi_arm", kwargs)

        # Outputs
        self.chip_depth = findDefault(0., "chip_depth", kwargs)
        self.F_radial = findDefault(0., "F_radial", kwargs)
        self.torque_cut = findDefault(0., "torque_cut", kwargs)

        # Calculated Values
        self.blade_center = self.saw.bladePosition()

        # Calibration Constants
        self.a0 = findDefault(0., "a0", kwargs)
        self.a1 = findDefault(0., "a1", kwargs)
        self.a2 = findDefault(0., "a2", kwargs)
        self.a3 = findDefault(0., "a3", kwargs)
        self.a4 = findDefault(0., "a4", kwargs)
        self.b0 = findDefault(0., "b0", kwargs)
        self.b1 = findDefault(0., "b1", kwargs)
        self.b2 = findDefault(0., "b2", kwargs)
        self.b3 = findDefault(0., "b3", kwargs)
        self.b4 = findDefault(0., "b4", kwargs)
        temp_K = self.calcPressureCoef(self.a0, self.a1, self.a2, self.a3, self.a4)
        self.K_fric = findDefault(temp_K, "K_fric", kwargs)
        temp_K = self.calcPressureCoef(self.b0, self.b1, self.b2, self.b3, self.b4)
        self.K_norm = findDefault(temp_K, "K_norm", kwargs)

    def calibrate(self, **kwargs):
        for key, arg in kwargs:
            if key == "a0": self.a0 = arg
            elif key == "a1": self.a1 = arg
            elif key == "a2": self.a2 = arg
            elif key == "a3": self.a3 = arg
            elif key == "a4": self.a4 = arg
            elif key == "b0": self.b0 = arg
            elif key == "b1": self.b1 = arg
            elif key == "b2": self.b2 = arg
            elif key == "b3": self.b3 = arg
            elif key == "b4": self.b4 = arg

    def calcPressureCoef(self, c0, c1, c2, c3, c4):
        """Calculates the pressure coefficients for use in Merchant's model using the regression 
        model found in Kapoor et al. (2024)."""
        temp = c0 + c1*np.log(self.chip_depth) + c2*np.log(self.V) + c3*self.alpha_n + c4*np.log(self.V)*np.log(self.chip_depth)
        return np.exp(temp)
    
    def calcCutArea(self):
        """Returns the area of the chip cross section."""
        t_c = self.chip_depth
        return t_c * self.saw.blade.kerf
    
    def setInputs(self):
        pass

    def definePath(self):
        pass
    
    def __str__(self):
        """Returns a string describing the object."""
        return "Cut (ID=" + str(self.id) + ")"

    def step(self):
        """Finds the intersection between the blade and workpiece, calculating the average
        radial height of the intersection, and then restructures the workpiece path to be the
        same as the blade circle where intersecting (removing cut material)."""
        self.blade_center = self.saw.bladePosition()
        intx_pts, seg_indices = self.findBladeWkpIntxs()
        if intx_pts is None:
            self.chip_depth = 0
            return
        self.chip_depth = self.calcAverageChipDepth(intx_pts, seg_indices)
        self.restructurePath(intx_pts, seg_indices)

    def findBladeWkpIntxs(self):
        """Finds the intersection points between the saw blade in its current position 
        and the workpiece. The blade will enter the wkp every odd point and exit every 
        even point."""
        intx_pts = list()
        seg_indices = list()
        for seg in self.wkp.path:
            points = self.findBladeSegIntxs(seg)
            for p in points:
                if p is not None: 
                    intx_pts.append(p)
                    seg_indices.append(self.wkp.path.index(seg))
        if len(intx_pts) < 2: return None, None
        self.arrangeIntxPointsByInOut(intx_pts, seg_indices)
        return intx_pts, seg_indices
    
    def findBladeSegIntxs(self, seg):
        """Finds the intersection points between the saw blade and a given line or arc segment."""
        if len(seg) < 3:
            points = geo.findCircleLineIntersections(seg[0], seg[1], center=self.blade_center, radius=self.saw.blade.radius)
        else:
            points= geo.findCircleArcIntersections(seg[0], seg[1], seg[2], center=self.blade_center, radius=self.saw.blade.radius)
        return geo.checkIfPointsOnSegment(points, seg=seg)
    
    def arrangeIntxPointsByInOut(self, points, seg_indices):
        """Returns the points arranged CCW so that odd points are entry points and even points
        are exit points for the blade passing through the wkp."""
        if self.checkCircleInProfile(self.blade_center, self.saw.blade.radius, 
                             points[0], points[1], seg_indices[0], seg_indices[1]):
            points.insert(0, points[-1])
            del(points[-1])
            seg_indices.insert(0, seg_indices[-1])
            del(seg_indices[-1])
        return points

    def checkCircleInProfile(self, center, radius, iP1, iP2, seg1_index, seg2_index) -> bool:
        """
        Checks if the given circle with given center and radius passes (CCW) inside a segmented 
        boundary curve between the intersection points iP1 and iP2, which lie on the two provided 
        segments found at self.wkp.path[seg1_index] and self.wkp.path[seg2_index].
        
        Process
        -------
        For any two consecutive points, P1, and P2:
        1. If P2 is not on the same segment as P1, then check the next vertex of the segment with
        P1.
        1. If P2 is on the same segment as P1, then parameterize the segment to w(t), so that 
        w(0) = P1 and w(1) = P2.
            1. Parameterize line: x = t * (P2(x) - P1(x)) + P1(x)
                                y = t * (P2(y) - P1(y)) + P1(y)
            1. Parameterize arc: x = Rcos(t*(theta(P2)-theta(P1) + theta(P1)) + C(x)
                                y = Rsin(t*(theta(P2)-theta(P1) + theta(P1)) + C(y)
        1. Check if w(k) for 0<k<1 is in the circle by seeing if the distance (D) from w(k) to the
        center of the blade circle is less than the radius (R) of the blade circle
        1. If D < R: then P1 is entry and P2 is exit (iff the blade moves from P1 to P2)

        Note that for t = 0.5, the parameterizations reduce to .5 * (P1 + P2)
        """
        seg1 = self.wkp.path[seg1_index]
        if seg1_index != seg2_index: check_pt = seg1[1]
        elif len(seg1) == 2: #segment line that intersects the primary arc twice
            check_pt = [.5 * (iP2[a] + iP1[a]) for a in range(2)]
        else: #segment arc that intersects the primary arc twice
            seg_cntr = geo.calcCircleCenter(*seg1)
            check_pt = geo.generatePointOnArc(seg1[0], seg1[1], seg_cntr)
        D = np.sqrt((check_pt[0] - center[0])**2 + (check_pt[1] - center[1])**2)
        return D < (radius - np.finfo(float).eps) 

    def calcAverageChipDepth(self, intx_pts, seg_indices) -> float:
        """Construed as the average of the distance between the blade circle and profile
        path."""
        profile_heights = list()
        for p in range(len(intx_pts) // 2): #Index to each odd (entry) intersection point
            profile = self.profile2PolarDomain(intx_pts[p:p+2], seg_indices[p:p+2])
            profile_heights.append(self.calcRadialDistanceByPoints(profile))
        return np.mean(profile_heights)

    def profile2PolarDomain(self, intx_pts, seg_indices):
        """Returns a 2D list detailing a closed segment path intersecting the blade circle.
        Each entry in the list is of the form: [<polar segment lambda>, <minimum bounding angle>,
        <maximum bounding angle>]."""
        profile = list()
        segs = self.getSegmentsInsideCircle(seg_indices)
        segs[0] = self.clipSegmentsOutsideCircle(segs[0], intx_pts[0])
        segs[-1] = self.clipSegmentsOutsideCircle(segs[-1], intx_pts[1])
        for seg in segs:
            entry = [geo.seg2PolarFun(seg, self.blade_center)]
            entry.extend(geo.calcBoundingAngles(seg, self.blade_center))
            profile.append(entry)
        return profile
    
    def getSegmentsInsideCircle(self, seg_indices):
        """Returns the two segments listed by seg_indices as well as all segments between
        which are enclosed within the circle."""
        n, m = seg_indices[0:2]
        check_pt = self.wkp.path[(n+1) % len(self.wkp.path)][0]
        isPos = geo.checkPointInCircle(check_pt, self.blade_center, self.saw.blade.radius)
        dir = [1, -1][isPos]
        segs = [self.wkp.path[n]]
        i = m
        while i != n:
            segs.append(self.wkp.path[i])
            i = (i + dir) % len(self.wkp.path)
        return segs

    def clipSegmentsOutsideCircle(self, seg, intx_pt, outside: bool=False):
        """Clip the portion of the segments found outside blade circle on a single side, biased
        towards the first vertex."""
        index = not geo.checkPointInCircle(seg[0], self.blade_center, self.saw.blade.radius)
        #Edited ^not^ up there, haven't proofed it yet.
        if outside: index = not index
        seg[index] = intx_pt
        return seg
    
    def calcRadialDistanceByPoints(self, profile, n: int=100) -> float:
        """
        Calculates the average distance between a profile and the blade circle.
        
        Process
        -------
        1. Profile segments are parameterized in terms of an angle, theta.
        1. Each segment is bounded by the minimum and maximum angular coordinate that evaluates 
        to the segment.
        1. The segments are sorted by minimum angular coordinate
        1. A list of n thetas is produced at which to check the radial distance of each segment 
        from the blade center.
        1. For each theta: 
            1. The list of segments is sliced to include all segments that have a radial value 
            at the theta in question.
            1. A list is made of each radial value for all sliced segments at the angle theta.
            1. If the list of points is odd, then the profile is bounded by the blade circle at 
            theta. The radial value for the blade (the blade radius) is added to the list.
            1. The distance between each point and it's successive neighbor is summed and added to
            a list of heights at each theta.
        1. The function returns the average of the list of heights.
        """
        h = list()
        profile = sorted(profile, key=lambda l : l[1])
        theta_min = profile[0][1]
        theta_max = max([a[2] for a in profile])
        for theta in np.linspace(theta_min, theta_max, n): #Check if theta is in range
            domain = [p for p in profile if p[2] > theta and p[2] < theta]
            h.append(self.findRadialHeightAtTheta(domain, theta))
        return np.mean(h)

    def findRadialHeightAtTheta(self, profile, theta) -> float:
        """Finds the total height of the chip at the angle theta."""
        if len(profile) == 0: return self.saw.blade.radius #no intersecting line, bounded by saw
        rs = list() #List of radial heights of the profile at theta
        for a in profile: rs.extend(a[0](theta))
        if len(rs) % 2 == 1: rs.append(self.saw.blade.radius)
        return sum([abs(rs[i+1] - rs[i]) for i in range(len(rs) // 2)])

    def restructurePath(self, intx_pts, seg_indices):
        """Deletes each segment in the workpiece path identified as starting at an odd index in
        seg_indices and ending at the following even index in seg_indices. Replaces these segments
        with a single arc that starts at the corresponding intersection points and is centered at the
        saw blade. Clips intersecting segments to intersection points."""
        num_segs = len(self.wkp.path)
        for p in range(len(intx_pts) // 2):
            p_i_pts = intx_pts[p:p+2]
            n, m = seg_indices[p:p+2]

            # Check if point on next segment is in the blade circle. If so, you can figure out the direction.
            # u -> n, m -> v (CCW)
            u, v = ( (n+1) % num_segs, (m-1) % num_segs )
            if geo.checkPointInCircle(self.wkp.path[u][0], self.blade_center, self.saw.blade.radius):
                u, v = ( (n-1) % num_segs, (m+1) % num_segs )

            # Clip intersecting segments
            #TODO: If the blade intersects a single segment, split the segment into two parts
            n_seg = [p_i_pts[0], self.wkp.path[u][0]]
            m_seg = [self.wkp.path[v][1], p_i_pts[1]]
            
            self.wkp.path[n] = n_seg
            if n == m:
                self.wkp.path.insert((m)%len(self.wkp.path), m_seg)
            else:
                self.wkp.path[m] = m_seg

            #Delete any enclosed segments
            if n != m:
                # Figure out which direction is the next enclosed segment
                next_pt = self.wkp.path[(n+1) % num_segs][0]
                isPos = geo.checkPointInCircle(next_pt, self.blade_center, self.saw.blade.radius)
                dir = [-1, 1][isPos]
                i = (n + dir) % num_segs
                while i != m:
                    del(self.wkp.path[i])
                    if not isPos: i -= 1
                    i = i % len(self.wkp.path) #increments automatically since len(path) decreases

            # Insert arc at n
            point_on_arc = geo.generatePointOnArc(*p_i_pts, self.blade_center)
            arc = [p_i_pts[1], p_i_pts[0], point_on_arc]
            arc_start_i = [pt[0] for pt in self.wkp.path].index(arc[1])
            self.wkp.path.insert(arc_start_i, arc)

