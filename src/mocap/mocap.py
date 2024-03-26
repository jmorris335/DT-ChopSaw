"""
| File: mocap.py 
| Info: Presents a class for performing direct linear transformation procedures for
|       basic motion capture.
| Author: John Morris, jhmrrs@clemson.edu  
| Organization: Product Lifecycle Management Center at Clemson University, plmcenter@clemson.edu  
| Permission: Copyright (C) 2023, John Morris. All rights reserved. Should not be reproduced, 
|       edited, sourced, or utilized without written permission from the author or organization
| Sources:
|   1. Rosebrock, A. (29 Sep 2014). "Finding the Brightest Spot in an Image using Python and 
|       OpenCV". PyImageSearch. Accessed 4 Mar 2024. https://pyimagesearch.com/2014/09/29/
|       finding-brightest-spot-image-using-python-opencv/
| Version History:
| - 0.0, 4 Mar 2024: initialized
"""

import numpy as np
import cv2

from src.auxiliary.geometry import pointDistance

class Mocap():
    '''
    Performs linear transformations of planar points into 3D space.

    Source: Kwon, Y. (1998). Direct Linear Transformation. Kwon3D. 
        Accessed 4 Mar 2024. http://www.kwon3d.com/theory/dlt/dlt.html
    '''

    def __init__(self, skeleton: list, num_cameras: int=2):
        """
        Parameters
        ----------
        skeleton : list
            A list of tuples (length 2) marking known locations of the skeleton points. The 
            points are solved in the order of this list.
        """
        self.is_calibrated = False
        self.global_skeleton = skeleton
        self.planar_skeleton = self.planar2global(self.global_skeleton)
        self.num_cameras = num_cameras
        
    def calibrate(self, control_pts):
        '''Calibrates the class for according to a calibration object for use in arbitrary
        coordination.
        
        Paramters
        ---------
        control_pts : array_like
            A list of a least 6 control points, where each entry is an (x, y, z) array.
        '''
        self.L = np.ones((1, 16))
        self.is_calibrated = True
        pass

    def planar2global(self, u, v) -> tuple:
        """Transforms the 2D point (u, v) from planar coordinates to global (3D) coordinants 
        (x, y, z)."""
        #TODO: Make a function for inverting Eq. 22 at http://www.kwon3d.com/theory/dlt/dlt.html
        pass

    def calcTransformationMatrix(self, planar_control_pts, global_control_pts) -> np.NDArray:
        """Calculates the transformation matrix for the camera system.

        The transformation matrix `P` maps all the points from `global_control_pts` to the equivalent
        points in `planar_control_pts`. After being calculated, assuming the cameras are not moved from
        their calibrated positions, global points can be calculated as `np.linalg(P, x)`, where `x` is
        a vector of the planar position coordinates.
        
        Parameters
        ----------
        planar_control_pts : 1xn vector of 1x2 vectors, where each subvector is an planar point 
            with coordinates (u, v)
        global_control_pts : 1xn vector of 1x3 vectors, where each subvector is a global point
            with coordinates (u, v, w). The point at each index must pair with the similar point
            in planar_control_pts

        Notes
        -----
        1. The number of control points (the length of `planar_control_pts` and `global_control_pts`, 
        n) must be at least 6.
        1. The control points cannot lie in the same plane (to avoid a singular system decomposition).
        """
        for c in range(self.num_cameras):
            # R = self.L9[c] * 
            pass

    def global2planar(self, x, y, z) -> tuple:
        """Transforms the global (3D) coordinates (x, y, z) to planar coordinants (u, v)."""
        denom = self.L9 * x + self.L10 * y + self.L11 * z + 1
        u = (self.L1 * x + self.L2 * y + self.L3 + self.L4) / denom
        v = (self.L5 * x + self.L6 * y + self.L7 + self.L8) / denom
        return u, v

    def updateSkeleton(self, img_path: str):
        """Updates the skeleton points to the points found in the image."""
        markers = self.getMarkerCoords(img_path, self.planar_skeleton)
        matches = self.matchMarkersWithSkeleton(markers)
        for i, sk in enumerate(self.planar_skeleton):
            self.planar_skeleton[i] = matches[sk]
        self.global_skeleton = self.planar2global(self.planar_skeleton)

    def getMarkerCoords(self, img_path: str, prev_markers: list) -> list:
        """Returns a list of planar coordinates for the brightest point around each 
        marker.
        
        Parameters
        ----------
        img_path : str
            Filepath for image to decode markers from.
        prev_markers : list
            2D list of planar coordinates for all markers previously identified.

        Returns
        -------
        list : list of calculated points for each marker updating the position in 
            `prev_markers`. The size is the same as `prev_markers`.
        """
        if prev_markers is None: 
            prev_markers = self.getDefaultMarkers()
        img = self.getImage(img_path)
        grayscale = self.makeGrayscale(img)
        blurred = self.blurImage(grayscale)
        new_pts = list()
        for marker in prev_markers:
            new_pt = self.findBrightestPoint(blurred, marker)
            new_pts.append(new_pt)
        return new_pts
    
    def matchMarkersWithSkeleton(self, markers: list) -> dict:
        """Returns a dictionary where each skeleton point is matched with a discovered
        marker point.

        Parameters
        ----------
        markers : list
            A list of possible coordinates for each skeleton point. Must be of greater length than 
            `self.planar_skeleton`.

        Returns
        -------
        dict : Dictionary where each key is a planar skeleton point (tuple) corresponding to the matched
            marker.

        Raises
        ------
        Exception
        """
        if len(markers) < len(self.planar_skeleton): 
            raise ValueError(f"Not enough markers ({len(markers)}) to match with Skeleton ({len(self.planar_skeleton)}).")
        matches = dict()
        for p_sk in self.planar_skeleton:
            d = [pointDistance(p_sk, m) for m in markers]
            marker = markers[d.index(min(d))]
            matches[p_sk] = marker
            markers.remove(marker)
        return matches

    def getDefaultMarkers(self) -> list:
        """Returns list of planar coordinates of the default markers for the saw in 
        resting position."""
        pass

    def getImage(self, img_path):
        """Returns the image at the filepath as an array_like object."""
        img = cv2.imread(img_path)
        return img
    
    def makeGrayscale(self, img):
        """Removes color in the image leaving only shades of black, grey, and white."""
        out = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return out
    
    def blurImage(self, img, radius=15):
        """Performs Gaussian blurring on the image with given radius."""
        blurred = cv2.GaussianBlur(img, (radius, radius), 0)
        return blurred
    
    def findBrightestPoint(self, img, point: tuple=None, dx: float=0.05, dy: float=None) -> tuple:
        """Returns the location for the brightest pixel in the image.
        
        Parameters
        ----------
        img : array_like
            The image to find the brightest pixel within.
        point : tuple, Optional
            Center of region to search for brightest pixel. If None, the method searches
            whole image.
        dx : float=0.5
            The width of the region to search as a proportion of the width of the image. Only 
            considered if `point` is passed as an parameter.
        dy : float, Optional
            The height of the region to search as a proportion of the height of the image.
            If not passed, dy is assumed to be equal to dx.
        
        Returns
        -------
        tuple : (x_coordinate, y_coordinate) specifying location of brightest pixel.

        Notes
        -----
        For region based searching the region is clipped by the image frame, so that regions 
        located near the edges of the image are necessarily smaller than indicated.
        """
        if point is not None:
            img = self.getRegionFromImage(img, point, dx, dy)
        (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(img)
        return (maxLoc.x, maxLoc.y)
        
    def getRegionFromImage(self, img, point: tuple, dx: float=0.05, dy: float=None):
        """Clips the given image to a box region centered at the `point` with width dx
        and height dy."""
        width = min([len(i) for i in img])
        height = len(img)
        dx = np.round(dx * width)
        if dy is None: 
            dy = dx
        dy = np.round(dy * height)

        min_x = np.rint(max(0, point[0] - dx)).astype(int)
        max_x = np.rint(min(width, point[0] + dx)).astype(int)
        min_y = np.rint(max(0, point[1] - dy)).astype(int)
        max_y = np.rint(min(width, point[1] + dy)).astype(int)
        clipped_img = img[min_x : max_x][min_y : max_y]

        return clipped_img


    

