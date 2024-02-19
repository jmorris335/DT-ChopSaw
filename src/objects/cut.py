"""
| File: cut.py 
| Info: Presents the state-model for a cut made with a chopsaw on a homogeneous workpiece 
| Author: John Morris, jhmrrs@clemson.edu  
| Organization: Product Lifecycle Management Center at Clemson University, plmcenter@clemson.edu  
| Permission: Copyright (C) 2023, John Morris. All rights reserved. Should not be reproduced, edited, sourced, or utilized without written permission from the author or organization
| Version History:
| - 0.1, 30 Jan 2024: Minimum viable functionality
| - 0.2, 15 Feb 2024: Full clipping functionality
"""

import numpy as np

from src.auxiliary.support import findDefault, loopSlice
import src.auxiliary.geometry as geo
from db.logger import Logger
from src.objects.twin import Twin
from src.objects.saw import Saw
from src.objects.workpiece import Workpiece

class Cut(Twin):
    '''
    Models the act of cutting a workpiece with a saw blade.

    Parameters
    ----------
    saw : Saw, default=Saw()
        The saw performing the cut, constructs a new ChopSaw object by default.
    wkp : Workpiece, default=Workpiece()
        The workpiece being cut, constructs a new Workpiece object by default.
    **kwargs : dict, optional
        Optional editing of state variables during initialization. Optional parameters are:
        a0 : float, default=0.0
            Calibration constant for tangential (fricational) loading.
        a1 : float, default=1.0
            Calibration constant for tangential (fricational) loading.
        a2 : float, default=1.0
            Calibration constant for tangential (fricational) loading.
        a3 : float, default=1.0
            Calibration constant for tangential (fricational) loading.
        a4 : float, default=1.0
            Calibration constant for tangential (fricational) loading.
        b0 : float, default=0.0
            Calibration constant for normal (radial) loading.
        b1 : float, default=1.0
            Calibration constant for normal (radial) loading.
        b2 : float, default=1.0
            Calibration constant for normal (radial) loading.
        b3 : float, default=1.0
            Calibration constant for normal (radial) loading.
        b4 : float, default=1.0
            Calibration constant for normal (radial) loading.

    References
    ----------
    1. Kapoor, S. G., R. E. DeVor, R. Zhu, R. Gajjela, G. Parakkal, and D. Smithey. “Development of Mechanistic Models for the Prediction of Machining Performance: Model Building Methodology.” Machining Science and Technology 2, no. 2 (December 1, 1998): 213-38. https://doi.org/10.1080/10940349808945669.
    '''
    def __init__(self, saw: Saw=None, wkp: Workpiece=None, **kwargs):
        Twin.__init__(self, **kwargs)

        # Components
        self.saw = saw if saw is not None else Saw()
        self.wkp = wkp if wkp is not None else Workpiece()
        
        # Static Values
        self.name = findDefault("Cut", "name", kwargs)
        self.a0 = findDefault(0., "a0", kwargs)
        self.a1 = findDefault(1., "a1", kwargs)
        self.a2 = findDefault(1., "a2", kwargs)
        self.a3 = findDefault(1., "a3", kwargs)
        self.a4 = findDefault(1., "a4", kwargs)
        temp_K = self.calcPressureCoef(self.a0, self.a1, self.a2, self.a3, self.a4)
        self.K_fric = findDefault(temp_K, "K_fric", kwargs)

        self.b0 = findDefault(0., "b0", kwargs)
        self.b1 = findDefault(1., "b1", kwargs)
        self.b2 = findDefault(1., "b2", kwargs)
        self.b3 = findDefault(1., "b3", kwargs)
        self.b4 = findDefault(1., "b4", kwargs)
        temp_K = self.calcPressureCoef(self.b0, self.b1, self.b2, self.b3, self.b4)
        self.K_norm = findDefault(temp_K, "K_norm", kwargs)

        # Dynamic Values
        self.all_cut = False
        self.cut_depth = findDefault(0., "chip_depth", kwargs)
        self.blade_center = self.saw.bladePosition

        # Twin inherited methods/attributes overloading
        self.log = Logger(self)
        self.objects = [self.saw, self.wkp]
        self.patches = self.saw.patches + self.wkp.patches

    def step(self):
        """Finds the intersection between the blade and workpiece, calculating the average
        radial height of the intersection, and then restructures the workpiece path to be the
        same as the blade circle where intersecting (removing cut material)."""
        if self.all_cut: return
        self.blade_center = self.saw.bladePosition

        if not self.all_cut:
            if self.checkWkpEnclosedInBlade():
                self.wkp.loops = list()
                self.updatePatches()
                self.all_cut = True
                self.cut_depth = 0.0
            else:
                self.performCutOperation()

        self.updateDynamics()
        Twin.step(self)

    def calcNormalPressureCoef(self):
        """Calculates the pressure coefficients for the normal force on the rake face."""
        return self.calcPressureCoef(self.a0, self.a1, self.a2, self.a3, self.a4)

    def calcTangentialPressureCoef(self):
        """Calculates the pressure coefficients for the normal force on the rake face."""
        return self.calcPressureCoef(self.b0, self.b1, self.b2, self.b3, self.b4)

    def calcPressureCoef(self, c0, c1, c2, c3, c4):
        """Calculates the pressure coefficients for use in Merchant's model using the regression 
        model found in Kapoor et al. (2024)."""
        V = abs(self.saw.blade.omega_blade)
        lnK = c0 + c1*np.log(self.cut_depth) + c2*np.log(V) + c3*self.saw.blade.rake + c4*np.log(V)*np.log(self.cut_depth)
        if np.isnan(lnK):
            return 0
        return np.exp(lnK)
    
    def calcCutArea(self):
        """Returns the area of the chip cross section."""
        t_c = self.cut_depth
        return t_c * self.saw.blade.kerf_blade
    
    def calcTangentForce(self):
        """Returns the force tangent the to cutting tool."""
        K = self.calcNormalPressureCoef()
        A = self.calcCutArea()
        return -K*A
    
    def updateDynamics(self):
        """Updates the inputs for each object."""
        cut_torque = self.calcTangentForce() * self.saw.blade.radius_blade
        self.log.addData("cut_torque", cut_torque)
        self.saw.blade.torque += cut_torque

    def performCutOperation(self):
        """Checks the workpiece and blade circle for intersections and clips the resulting 
        workpiece profiles."""
        chip_depths = list()
        for i in range(len(self.wkp.loops)):
            loop = self.wkp.loops[i]
            if self.checkLoopEnclosedInBlade(loop):
                self.wkp.loops[i] = list()
                continue
            if self.checkBladeOutsideLoop(loop):
                continue
            intx_pts, seg_indices = self.findBladeWkpIntxs(loop)
            if intx_pts is None:
                continue
            loop_chip_depth = self.calcAverageChipDepth(intx_pts, seg_indices, loop)
            chip_depths.append(loop_chip_depth)
            self.wkp.loops[i] = self.restructurePath(intx_pts, seg_indices, loop)
        if len(chip_depths) > 0:
            self.log.setData("cut_depth", np.mean(chip_depths))
        self.wkp.loops = self.wkp.cleanLoops()
        self.updatePatches()

    def checkWkpEnclosedInBlade(self) -> bool:
        """Returns true if the workpiece is entirely enclosed inside the blade circle 
        (indicating a complete cut)."""
        for loop in self.wkp.loops:
            if not self.checkLoopEnclosedInBlade(loop):
                return False
        self.log.log("Workpiece enclosed in blade.")
        return True
    
    def checkLoopEnclosedInBlade(self, loop) -> bool:
        """Returns True if the loop is entirely enclosed inside the blade circle. The blade 
        circle is allowed to intersecting within the tolerance of the workpiece."""
        for seg in loop:
            seg_dist = geo.pointSegDistance(self.blade_center, seg, is_max=True)
            if seg_dist - self.saw.blade.radius_blade > -self.wkp.tol:
                return False
        return True
    
    def checkBladeOutsideLoop(self, loop) -> bool:
        """Returns True if the blade circle does not intersect (or only tangentially 
        intersects) the loop. Fails if the blade is completely enclosed within the loop.
        The blade is allowed to be as close as the tolerance of the workpiece."""
        for seg in loop:
            seg_dist = geo.pointSegDistance(self.blade_center, seg, is_max=False)
            if seg_dist - self.saw.blade.radius_blade < -self.wkp.tol:
                return False
        return True

    def findBladeWkpIntxs(self, loop: list):
        """Finds the intersection points between the saw blade in its current position 
        and the workpiece. The blade will enter the wkp every odd point and exit every 
        even point."""
        intx_pts = list()
        seg_indices = list()
        for seg in loop:
            points = self.findBladeSegIntxs(seg)
            for p in points:
                if p is not None: 
                    intx_pts.append(p)
                    seg_indices.append(loop.index(seg))
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
        that point is in the workpiece path. If it is, then the points are arranged In/Out.
        """
        if CW:
            check_pt = geo.generatePointOnArc(P2, P1, self.blade_center)
        else:
            check_pt = geo.generatePointOnArc(P1, P2, self.blade_center)
        return self.wkp.pointInLoop(check_pt)

    def calcAverageChipDepth(self, intx_pts, seg_indices, loop: list) -> float:
        """Construed as the average of the distance between the blade circle and path
        path."""
        path_heights = list()
        for p in range(0, len(intx_pts), 2): #Index to each odd (entry) intersection point
            path = self.path2PolarDomain(intx_pts[p:p+2], seg_indices[p:p+2], loop)
            path_heights.append(self.calcRadialDistanceByPoints(path))
        return np.mean(path_heights)

    def path2PolarDomain(self, intx_pts, seg_indices, loop: list) -> list:
        """Returns a 2D list detailing a closed segment path intersecting the blade circle.
        Each entry in the list is of the form: [<polar segment lambda>, <minimum bounding angle>,
        <maximum bounding angle>]."""
        path = list()
        enclosed_path = self.getSegmentsInsideCircle(intx_pts, seg_indices, loop)
        segs = [p[:] for p in enclosed_path]

        # Clip interior segments
        is_pos = self.loopIndexingIsPos(intx_pts, seg_indices, loop)
        segs[0] = self.clipSegment(segs[0], intx_pts[0], is_pos, is_in=True, external=False)
        segs[-1] = self.clipSegment(segs[-1], intx_pts[-1], is_pos, is_in=False, external=False)
        
        for seg in segs:
            entry = [geo.seg2PolarFun(seg, self.blade_center)]
            entry.extend(geo.calcBoundingAngles(seg, self.blade_center))
            path.append(entry)
        return path
    
    def getSegmentsInsideCircle(self, intx_pts, seg_indices, loop: list) -> list:
        """Returns the two segments listed by seg_indices as well as all segments between
        which are enclosed within the circle."""
        n, m = seg_indices[0:2]
        if self.loopIndexingIsPos(intx_pts, seg_indices, loop):
            return loopSlice(loop, m, n+1)
        else:
            return loopSlice(loop, n, m+1)
    
    def findVertexByIntxPt(self, pt, seg: list) -> list:
        """Returns a vertex in the loop that whose euclidean distance to the given point,
        pt, is less than the workpiece tolerance. If there are no vertices within that distance, 
        returns None."""
        tol = self.wkp.tol
        for vtx in seg[0:2]:
            if geo.pointDistance(pt, vtx) < tol:
                return vtx
        return None
    
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
        for a in profile: 
            rs.extend(a[0](theta))
        if len(rs) % 2 == 1: #chip terminated by blade
            rs.append(self.saw.blade.radius_blade)
        return sum([abs(rs[i+1] - rs[i]) for i in range(len(rs) // 2)])

    def restructurePath(self, intx_pts, seg_indices, loop: list):
        """Deletes each segment in the workpiece path identified as starting at an odd index in
        seg_indices and ending at the following even index in seg_indices. Replaces these segments
        with a single arc that starts at the corresponding intersection points and is centered at the
        saw blade. Clips intersecting segments to intersection points.
        """
        is_pos = self.loopIndexingIsPos(intx_pts, seg_indices, loop)
        clipped_segs = self.getClippedLoopSegments(intx_pts, seg_indices, is_pos, loop)
        new_path = self.makeNewPath(seg_indices, clipped_segs, is_pos, loop)
        
        p_idxs = [i*2 for i in range(len(intx_pts) // 2)]
        for p in p_idxs:
            self.insertCutArc(*intx_pts[p:p+2], is_pos, new_path)
        return new_path
    
    def getClippedLoopSegments(self, intx_pts, seg_indices, is_pos: bool, loop: list) -> list:
        """Returns a list of the clipped segments that have intersection points. The clipped
        segments are the segments exterior to the blade circle (remaining after the cut) and 
        are indexed by the seg_indices array in the passed loop. 
        
        Note that the intersection points may be updated as a result of the function.
        """
        clipped_segs = list()
        for p in range(len(intx_pts)):
            idx = seg_indices[p]
            #Check if intx_pt is too close to loop vertex to clip.
            close_vtx = self.findVertexByIntxPt(intx_pts[p], loop[seg_indices[p]])
            if close_vtx is not None: 
                c_s = loop[idx]
                intx_pts[p] = close_vtx
            else:
                is_in = p % 2 == 0
                c_s = self.clipSegment(loop[idx], intx_pts[p], is_pos, is_in)

            clipped_segs.append(c_s)
        return clipped_segs
    
    def clipSegment(self, seg, pt, is_pos: bool, is_in: bool, external: bool=True):
        """Returns the clipped segment that must start at pt (an intersection point) and end at
        either endpoint in the segment (seg).

        Parameters
        ----------
        seg : array_like
            The segment to clip, either 1x2 or 1x3 array of 1x2 point arrays.
        pt : array_like
            The intersection point to clip to, as a 1x2 point array.
        is_pos : bool
            Indicates that the workpiece loop is defined in a CCW method so that the a cut starts
            at segment i, and then passes through segments i+1...i+n-1 until reaching the out 
            segment.
        is_in : bool
            Indicates that the intersection point considered is an entry point for the blade
            versus and exit point.
        external : bool, default=True
            True if the returned segment should be the portion of the segment exterior to the 
            blade circle (remaining) or interior (removed).
        """
        possible_clippings = [[pt, seg[1]], [seg[0], pt]]
        if external != is_pos: possible_clippings.reverse()
        clipped_seg = possible_clippings[is_in^1]

        # Generate additional point for arcs
        if len(seg) == 3: 
            clipped_seg = self.generateClippedArcSeg(seg, clipped_seg)
        return clipped_seg
    
    def generateClippedArcSeg(self, arc_seg, endpoints):
        """Generates an arbitrary segement for an arc bounded by the endpoints and coincident
        with the arc segment."""
        arc_center = geo.calcCircleCenter(*arc_seg)
        if not geo.arcIsCCW(*arc_seg, arc_center):
            endpoints.append(geo.generatePointOnArc(*reversed(endpoints), arc_center))
        else:
            endpoints.append(geo.generatePointOnArc(*endpoints, arc_center))
        return endpoints

    def loopIndexingIsPos(self, intx_pts, seg_indices, loop: list):
        """Returns True if the segment indices for the loop starting at the point A on the 
        segment located at index i_in is positive or negative incrementing. That is, if the remaining 
        loop is composed of segments at [i_in, i_in+1, i_in+2, ..., i_in+k, i_out] or the segments at 
        [i_in, i_in-1, i_in-2, ..., i_in-k, i_out]."""
        i_in = seg_indices[0]
        pt_in = intx_pts[0]
        pt_indices = [i for i in range(len(seg_indices)) if seg_indices[i] == i_in]
        if len(pt_indices) >= 2: #2 intersections on segment
            A, B = [intx_pts[a] for a in pt_indices]
            pt_out = A if B == pt_in else B
            t_in = geo.calcParameterizedPointOnSeg(pt_in, loop[i_in])
            t_out = geo.calcParameterizedPointOnSeg(pt_out, loop[i_in])
            return t_in > t_out

        next_pt = loop[(i_in+1) % len(loop)][0]
        isPos = not geo.checkPointInCircle(next_pt, self.blade_center, self.saw.blade.radius_blade)
        return isPos

    def makeNewPath(self, seg_indices, clipped_segs, is_pos, loop: list):
        """Returns a new path containing all the segments between the indices i_in and i_out that
        are outside of the blade circle. The path needs to be cleaned to identify possible loops."""
        num_segs = len(loop)
        new_lp = list()

        i = seg_indices[is_pos]
        not_visited_ps = [p for p in range(is_pos, len(seg_indices), 2)] #all out/in segs
        while len(not_visited_ps) > 0:
            p = self.findNextClippedSegIdx(not_visited_ps, seg_indices, i)
            p, c_s = self.findNextClippedSegs(p, clipped_segs, is_pos)
            new_lp.extend(c_s)

            # Append non-intersecting, non-enclosed segments
            i = (seg_indices[p] + 1) % num_segs
            while i not in seg_indices:
                new_lp.append(loop[i])
                i = (i+1) % num_segs

        return new_lp
    
    def findNextClippedSegIdx(self, not_visited_ps, seg_indices, i):
        """Returns the index for the next clipped segment in the path, with the direction
        decided by the indexing of the loop. Deletes the index from the not_visited_ps
        list."""
        found = False
        for j in not_visited_ps:
            if seg_indices[j] == i:
                p = j
                found = True
        if not found: p = not_visited_ps[0]
        del(not_visited_ps[not_visited_ps.index(p)])
        return p
    
    def findNextClippedSegs(self, p: int, clipped_segs, is_pos: bool):
        """Returns the set of clipped segments in order based on the index of the next
        connecting segment, p. The connecting segment is out/in based on if is_pos is 
        True/False."""
        if is_pos:  #enclosed segs: out -> in
            c_s = [clipped_segs[p], clipped_segs[p-1]]
            p -= 1
        else:   #remaining segs: out -> in
            c_s = clipped_segs[p:p+2]
            p += 1
        return p, c_s

    def insertCutArc(self, A, B, is_pos: bool, loop: list):
        """Inserts an arc coicident with the blade circle and terminated by points A
        and B. Fails if the workpiece profile is crossed (not simple)."""
        point_on_arc = geo.generatePointOnArc(A, B, self.blade_center)
        if is_pos:
            arc = [B, A, point_on_arc]
        else:
            arc = [A, B, point_on_arc]
        arc_start_i = [pt[0] for pt in loop].index(arc[1])
        loop.insert(arc_start_i, arc)