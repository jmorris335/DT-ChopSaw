"""
| File: cut.py 
| Info: Presents the state-model for a cut made with a chopsaw on a homogeneous workpiece 
| Author: John Morris, jhmrrs@clemson.edu  
| Organization: Product Lifecycle Management Center at Clemson University, plmcenter@clemson.edu  
| Permission: Copyright (C) 2023, John Morris. All rights reserved. Should not be reproduced, edited, sourced, or utilized without written permission from the author or organization
| Version History:
| - 0.1, 30 Jan 2023: Minimum viable functionality
"""

import numpy as np

from src.aux.support import findDefault, loopSlice
import src.aux.geometry as geo
from src.objects.saw import Saw
from src.objects.workpiece import Workpiece

class Cut():
    '''
    Models the act of cutting a workpiece with a saw blade.

    Parameters
    ----------
    saw : Saw, default=Saw()
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
    def __init__(self, saw=Saw(), wkp=Workpiece(), **kwargs):
        self.id = findDefault("0", "id", kwargs)
        self.all_cut = False

        # Components
        self.saw = saw
        self.wkp = wkp

        # Physical Values
        self.alpha_n = findDefault(self.saw.blade.rake, "alpha_n", kwargs)
        # self.t_c = findDefault(self.saw.blade.kerf, "t_c", kwargs)

        # Dyanmic Values
        # self.h = findDefault(self.H, "h", kwargs)

        # Inputs
        self.V = findDefault(self.saw.blade.omega_blade*self.saw.blade.radius_blade, "V", kwargs)
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
        self.a1 = findDefault(1., "a1", kwargs)
        self.a2 = findDefault(1., "a2", kwargs)
        self.a3 = findDefault(1., "a3", kwargs)
        self.a4 = findDefault(1., "a4", kwargs)
        self.b0 = findDefault(0., "b0", kwargs)
        self.b1 = findDefault(1., "b1", kwargs)
        self.b2 = findDefault(1., "b2", kwargs)
        self.b3 = findDefault(1., "b3", kwargs)
        self.b4 = findDefault(1., "b4", kwargs)
        temp_K = self.calcPressureCoef(self.a0, self.a1, self.a2, self.a3, self.a4)
        self.K_fric = findDefault(temp_K, "K_fric", kwargs)
        temp_K = self.calcPressureCoef(self.b0, self.b1, self.b2, self.b3, self.b4)
        self.K_norm = findDefault(temp_K, "K_norm", kwargs)

        # GUI Operations
        self.objects = [saw, wkp]
        self.patches = saw.patches + wkp.patches

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
        lnK = c0 + c1*np.log(self.chip_depth) + c2*np.log(self.V) + c3*self.alpha_n + c4*np.log(self.V)*np.log(self.chip_depth)
        return np.exp(lnK)
    
    def calcCutArea(self):
        """Returns the area of the chip cross section."""
        t_c = self.chip_depth
        return t_c * self.saw.blade.kerf
    
    def __str__(self):
        """Returns a string describing the object."""
        return "Cut (ID=" + str(self.id) + ")"
    
    def set(self, **kwargs):
        """Updates parameters in the saw."""
        for object in self.objects:
            object.set(**kwargs)
    
    def updatePatches(self):
        self.saw.updatePatches()
        if not self.all_cut:
            self.wkp.updatePatches()

    def step(self):
        """Finds the intersection between the blade and workpiece, calculating the average
        radial height of the intersection, and then restructures the workpiece path to be the
        same as the blade circle where intersecting (removing cut material)."""
        if self.all_cut: return
        self.blade_center = self.saw.bladePosition()
        if self.checkWkpEnclosedInBlade(): #check if cut finished
            del(self.patches[1])
            self.all_cut = True
            return
        intx_pts, seg_indices = self.findBladeWkpIntxs()
        if intx_pts is None:
            self.chip_depth = 0
            return
        self.chip_depth = self.calcAverageChipDepth(intx_pts, seg_indices)
        self.restructurePath(intx_pts, seg_indices)
        for object in self.objects:
            object.step()

    def checkWkpEnclosedInBlade(self) -> bool:
        """Returns true if the workpiece is entirely enclosed inside the blade circle 
        (indicating a complete cut)."""
        for seg in self.wkp.path:
            seg_dist = geo.pointSegDistance(self.blade_center, seg)
            if seg_dist - self.saw.blade.radius_blade > -geo.eps:
                return False
        return True

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
        intx_pts, seg_indices = self.arrangeIntxPointsByInOut(intx_pts, seg_indices)
        return intx_pts, seg_indices
    
    def findBladeSegIntxs(self, seg):
        """Finds the intersection points between the saw blade and a given line or arc segment."""
        if len(seg) < 3:
            points = geo.findCircleLineIntersections(*seg, center=self.blade_center, radius=self.saw.blade.radius_blade)
        else:
            points= geo.findCircleArcIntersections(*seg, center=self.blade_center, radius=self.saw.blade.radius_blade)
        return geo.checkIfPointsOnSegment(points, seg=seg)
    
    def arrangeIntxPointsByInOut(self, intx_pts, seg_indices):
        """Returns the points arranged CCW so that odd points are entry points and even points
        are exit points for the blade passing through the wkp."""
        intx_pts, seg_indices = geo.sortPointsByTheta(intx_pts, self.blade_center, [seg_indices])
        if not self.checkIntxPointsAreInOut(*intx_pts[:2]):
            intx_pts = intx_pts[-1:] + intx_pts[:-1]
            seg_indices = seg_indices[-1:] + seg_indices[:-1]
        return intx_pts, seg_indices

    def checkIntxPointsAreInOut(self, P1, P2, CW=False) -> bool:
        """Determines if the sequential intersection points are arranged as In/Out or Out/In,
        where In indicates the point is where the blade enters the material, and Out is an exit 
        point. 
        
        Process
        -------
        Generates a point on the blade circle between the two intersection points and checks if 
        that point is in the workpiece profile. If it is, then the points are arranged In/Out.
        """
        if CW:
            check_pt = geo.generatePointOnArc(P2, P1, self.blade_center)
        else:
            check_pt = geo.generatePointOnArc(P1, P2, self.blade_center)
        return self.wkp.pointInPath(check_pt)
    
    # def checkIntxPointsAreInOut(self, P1, P2, seg1_idx, seg2_idx) -> bool:
    #     """Returns True if the blade circle contains the segments between the intersection
    #     points, indicating that the first intersection point is an entry point and the second
    #     is an exit point. Returns false if the profile is within the blade circle, indicating
    #     the opposite.
        
    #     Process
    #     -------
    #     Intersection points (when ordered by increasing angular distance from an axis) always come
    #     in pairs, either In/Out or Out/In. If the pair is In/Out (meaning the blade cuts into the 
    #     workpiece at the first intersection point) then any point on the profile between the two pionts
    #     must be outside of the blade circle, and vice versa for Out/In. So the process of determining
    #     order is tied to finding a point on the profile between the two intersection points and 
    #     checking if it is inside or outside of the blade circle. 
    #     """
    #     seg1 = self.wkp.path[seg1_idx]
    #     if seg2_idx > seg1_idx: #Check vertex following P1
    #         check_pt = seg1[1]
    #     elif seg1_idx > seg2_idx: #Check vertex preceding P1
    #         check_pt = seg1[0]
    #     else: #See if P1 comes before or after P2 on segment
    #         t_P1 = geo.calcParameterizedPointOnSeg(P1, seg1)
    #         t_P2 = geo.calcParameterizedPointOnSeg(P2, seg1)
    #         if t_P1 > t_P2: #Check vertex following P1
    #             check_pt = seg1[1]
    #         else: #Check point in between P1 and P2
    #             if len(seg1) < 3:
    #                 check_pt = [.5 * (P1[i] + P2[i]) for i in range(2)]
    #             else:
    #                 seg_cntr = geo.calcCircleCenter(*seg1)
    #                 check_pt = geo.generatePointOnArc(P1, P2, seg_cntr)
    #     return geo.checkPointInCircle(check_pt, self.blade_center, self.saw.blade.radius_blade)

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
        enclosed_path = self.getSegmentsInsideCircle(seg_indices)
        segs = [p[:] for p in enclosed_path]
        segs[0] = self.clipSegmentsOutsideCircle(segs[0], intx_pts[0])
        segs[-1] = self.clipSegmentsOutsideCircle(segs[-1], intx_pts[1])
        for seg in segs:
            entry = [geo.seg2PolarFun(seg, self.blade_center)]
            entry.extend(geo.calcBoundingAngles(seg, self.blade_center))
            profile.append(entry)
        return profile
    
    def getSegmentsInsideCircle(self, seg_indices) -> list:
        """Returns the two segments listed by seg_indices as well as all segments between
        which are enclosed within the circle."""
        n, m = seg_indices[0:2]
        check_pt = self.wkp.path[(n+1) % len(self.wkp.path)][0]
        isPos = geo.checkPointInCircle(check_pt, self.blade_center, self.saw.blade.radius_blade)
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
        index = not geo.checkPointInCircle(seg[0], self.blade_center, self.saw.blade.radius_blade)
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
        if len(profile) == 0: return self.saw.blade.radius_blade #no intersecting line, bounded by saw
        rs = list() #List of radial heights of the profile at theta
        for a in profile: rs.extend(a[0](theta))
        if len(rs) % 2 == 1: rs.append(self.saw.blade.radius_blade)
        return sum([abs(rs[i+1] - rs[i]) for i in range(len(rs) // 2)])

    def restructurePath(self, intx_pts, seg_indices):
        """Deletes each segment in the workpiece path identified as starting at an odd index in
        seg_indices and ending at the following even index in seg_indices. Replaces these segments
        with a single arc that starts at the corresponding intersection points and is centered at the
        saw blade. Clips intersecting segments to intersection points."""
        for p in range(len(intx_pts) // 2):
            p_i_pts = intx_pts[p:p+2]
            n, m = seg_indices[p:p+2]

            n_seg, m_seg = self.findClippedSegs(n, m, *p_i_pts)
            
            self.wkp.path[n] = n_seg 
            if n == m: #Make a new segment if both intersections are through a single segment
                self.wkp.path.insert(m, m_seg)
            else:
                self.wkp.path[m] = m_seg

            self.deleteEnclosedSegments(n, m)
            self.insertCutArc(*p_i_pts)

    def findClippedSegs(self, n, m, A, B):
        """Returns the clipped segment(s) intersecting the blade circle that remain 
        (not enclosed in the blade circle). A and B are the (x,y) intersection points located on the
        segments found at indices n and m.
        
        Note: function uses the values of the adjacent segments to connect lines together. These
        adjacent segments (located at indices u and v) are determined by finding which direction 
        the blade circle intersection occurs.
        """
        num_segs = len(self.wkp.path)
        u, v = ( (n+1) % num_segs, (m-1) % num_segs )
        if geo.checkPointInCircle(self.wkp.path[u][0], self.blade_center, self.saw.blade.radius_blade):
            u, v = ( (n-1) % num_segs, (m+1) % num_segs )

        # Clip intersecting segments
        n_seg = [A, self.wkp.path[u][0]]
        m_seg = [self.wkp.path[v][1], B]

        # Generate additional point for arcs
        if len(self.wkp.path[n]) == 3: 
            n_seg = self.generateClippedArcSeg(n, n_seg)
        if len(self.wkp.path[m]) == 3: 
            m_seg = self.generateClippedArcSeg(m, m_seg)
        return n_seg, m_seg

    def generateClippedArcSeg(self, index, endpoints):
        """Generates an arbitrary segement for an arc bounded by the endpoints and coincident
        with the arc segment located at self.wkp.path[index]."""
        arc_center = geo.calcCircleCenter(*self.wkp.path[index])
        if not geo.arcIsCCW(*self.wkp.path[index], arc_center):
            endpoints.append(geo.generatePointOnArc(*reversed(endpoints), arc_center))
        else:
            endpoints.append(geo.generatePointOnArc(*endpoints, arc_center))
        return endpoints
    
    def deleteEnclosedSegments(self, n, m):
        """Removes all segments between the segments at the indices n and m that 
        are entirely enclosed in the blade circle."""
        if n == m: return #No enclosed segments

        # Figure out which direction is the next enclosed segment
        num_segs = len(self.wkp.path)
        next_pt = self.wkp.path[(n+1) % num_segs][0]
        isPos = geo.checkPointInCircle(next_pt, self.blade_center, self.saw.blade.radius_blade)
        dir = [-1, 1][isPos]
        i = (n + dir) % num_segs
        while i != m:
            del(self.wkp.path[i])
            if not isPos: i -= 1
            i = i % len(self.wkp.path) #increments automatically since len(path) decreases

    def insertCutArc(self, A, B):
        """Inserts an arc coicident with the blade circle and terminated by points A
        and B. Fails if the workpiece profile is crossed (not simple)."""
        point_on_arc = geo.generatePointOnArc(A, B, self.blade_center)
        arc = [B, A, point_on_arc]
        arc_start_i = [pt[0] for pt in self.wkp.path].index(arc[1])
        self.wkp.path.insert(arc_start_i, arc)