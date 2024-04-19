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
import cv2 as cv

from src.mocap.calibration import calibrateCameraIntrinsic, stereoCalibration
from src.auxiliary.geometry import pointDistance
from src.db.actor import DBActor

class Mocap():
    """Performs linear transformations of planar points into 3D space.

    Source: Kwon, Y. (1998). Direct Linear Transformation. Kwon3D. 
        Accessed 4 Mar 2024. http://www.kwon3d.com/theory/dlt/dlt.html
    """

    def __init__(self, skeleton: list=None, num_cameras: int=2):
        """
        Parameters
        ----------
        skeleton : list
            A list of tuples (length 3) marking known global locations of the skeleton points. The 
            points are solved in the order of this list.

        Class Members
        -------------
        planar_skeletons : list
            A list (length equal to `num_cameras`) of planar coordinates of each skeleton point
            in the coordinate system of the camera
        """
        self.db = DBActor()
        self.is_calibrated = False
        self.global_skeleton = skeleton
        if skeleton is None:
            self.planar_skeletons = self.getDefaultMarkers()
        # self.planar_skeletons = self.planar2global(self.global_skeleton)
        self.num_cameras = num_cameras
        self.calibrate()

        self.markers = dict()
        self.initializeMarkers()

    def initializeMarkers(self, write2DB: bool=False):
        labels = ['base_1', 'base_2', 'miter_1', 'bevel_1', 'arm_1', 'arm_2']
        self.writeMarkers2DB(labels)

    def calibrate(self):
        """Calibrates the class for according to a calibration object for use in arbitrary
        coordination.
        
        Paramters
        ---------
        control_pts : array_like
            A list of a least 6 control points, where each entry is an (x, y, z) array.
        """
        cameras = ["A", "B"]
        num_rows = 4
        num_cols = 7
        square_size = 2.5 #Units are cm
        cam_mtrxs, cam_dist_coeffs = list(), list()
        for camera in cameras:
            # mono_img_paths = [f"src/mocap/test-mocap/{camera}_calib_{i+1}.jpg" for i in range(9)]
            mono_img_paths = [f"src/mocap/temugeB_demo/frames/mono_calib/camera{camera}_{i}.png" for i in range(4)]
            calib_consts = calibrateCameraIntrinsic(mono_img_paths, num_rows, num_cols, square_size, show_images=False)
            cam_mtrxs.append(calib_consts['camera_matrix'])
            cam_dist_coeffs.append(calib_consts['dist_coeffs'])
    
        stereo_img_paths_A = [f"src/mocap/temugeB_demo/frames/synched/cameraA_{i}.png" for i in range(4)]
        stereo_img_paths_B = [f"src/mocap/temugeB_demo/frames/synched/cameraB_{i}.png" for i in range(4)]
        rmse, R, T = stereoCalibration(stereo_img_paths_A, stereo_img_paths_B, *cam_mtrxs, *cam_dist_coeffs, 
                      num_rows, num_cols, square_size, show_images=False)

        self.projection_mtrxs = list()
        RT1 = np.concatenate([np.eye(3), [[0],[0],[0]]], axis = -1)
        P1 = R @ RT1 #projection matrix for C1
        self.projection_mtrxs.append(P1)
    
        RT2 = np.concatenate([R, T], axis = -1)
        P2 = R @ RT2 #projection matrix for C2
        self.projection_mtrxs.append(P2)

    def planar2global(self) -> tuple:
        """Transforms the 2D point (u, v) from planar coordinates to global (3D) coordinants 
        (x, y, z) using Direct Linear Transformation.

        Inputs are 2x1 array_like. Output is 3x2 array_like.
        
        #TODO: This is copied from temugeB as a placeholder, it needs to be reviewed and replaced.
        Currently returns the 3D coordinates for a point triangulated between 2 planar coordinate systems
        with given projection matrices. It also only works for 2 cameras."""
        global_pnts = list()
        u_pnts, v_pnts = self.planar_skeletons[:2]

        for u, v in zip(u_pnts, v_pnts):
            projection_mtrx_A = self.projection_mtrxs[0]
            projection_mtrx_B = self.projection_mtrxs[1]
            A = [u[1]*projection_mtrx_A[2,:] - projection_mtrx_A[1,:],
                projection_mtrx_A[0,:] - u[0]*projection_mtrx_A[2,:],
                v[1]*projection_mtrx_B[2,:] - projection_mtrx_B[1,:],
                projection_mtrx_B[0,:] - v[0]*projection_mtrx_B[2,:]
                ]
            A = np.array(A).reshape((4,4))
        
            B = A.transpose() @ A
            U, s, Vh = np.linalg.svd(B, full_matrices = False)
            global_pnts.append(Vh[3,0:3]/Vh[3,3])
        return global_pnts

    # def calcTransformationMatrix(self, planar_control_pts, global_control_pts) -> np.NDArray:
    #     """Calculates the transformation matrix for the camera system.

    #     The transformation matrix `P` maps all the points from `global_control_pts` to the equivalent
    #     points in `planar_control_pts`. After being calculated, assuming the cameras are not moved from
    #     their calibrated positions, global points can be calculated as `np.linalg(P, x)`, where `x` is
    #     a vector of the planar position coordinates.
        
    #     Parameters
    #     ----------
    #     planar_control_pts : 1xn vector of 1x2 vectors, where each subvector is an planar point 
    #         with coordinates (u, v)
    #     global_control_pts : 1xn vector of 1x3 vectors, where each subvector is a global point
    #         with coordinates (u, v, w). The point at each index must pair with the similar point
    #         in planar_control_pts

    #     Notes
    #     -----
    #     1. The number of control points (the length of `planar_control_pts` and `global_control_pts`, 
    #     n) must be at least 6.
    #     1. The control points cannot lie in the same plane (to avoid a singular system decomposition).
    #     """
    #     for c in range(self.num_cameras):
    #         # R = self.L9[c] * 
    #         pass

    # def global2planar(self, x, y, z) -> tuple:
    #     """Transforms the global (3D) coordinates (x, y, z) to planar coordinants (u, v)."""
    #     denom = self.L9 * x + self.L10 * y + self.L11 * z + 1
    #     u = (self.L1 * x + self.L2 * y + self.L3 + self.L4) / denom
    #     v = (self.L5 * x + self.L6 * y + self.L7 + self.L8) / denom
    #     return u, v

    def updateSkeleton(self, img_paths: list):
        """Updates the skeleton (planar and global) points to the points found in the image.
        
        img_paths is a list of paths of size similar to planar_skeletons.
        """
        for planar_skeleton, img_path in zip(self.planar_skeletons, img_paths):
            markers = self.getMarkerCoords(img_path, planar_skeleton)
            # planar_skeleton = self.matchMarkersWithSkeleton(markers, planar_skeleton)
            # self.displayMarkers(planar_skeleton, img_path)
            # self.displayMarkers(markers, img_path)
        self.global_skeleton = self.planar2global()
        self.marker2skeleton(self.global_skeleton)
        self.writeSequence2DB()

    def marker2skeleton(self, markers: list):
        for i in range(len(self.markers.keys())):
            key = list(self.markers.keys())[i]
            self.markers[key]['coords'] = markers[i]

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
        new_markers = list()
        for prev_marker in prev_markers:
            new_marker = self.findBrightestPoint(blurred, prev_marker)
            new_markers.append(new_marker)
        return new_markers
    
    def matchMarkersWithSkeleton(self, markers: list, prev_markers) -> dict:
        """Returns a dictionary where each skeleton point is matched with a discovered
        marker point.

        Parameters
        ----------
        markers : list
            A list of possible coordinates for each skeleton point. Must be of equal or greater 
            length than `self.planar_skeleton`.

        Returns
        -------
        dict : Dictionary where each key is a planar skeleton point (tuple) corresponding to the matched
            marker.

        Raises
        ------
        Exception
        """
        if len(markers) < len(prev_markers): 
            raise ValueError(f"Not enough markers ({len(markers)}) to match with Skeleton ({len(prev_markers)}).")
        matches = list()
        #Find which marker is closest to a skeleton point
        for p_sk in prev_markers:
            d = [pointDistance(p_sk, m) for m in markers]
            marker = markers[d.index(min(d))]
            matches.append(marker)
            markers.remove(marker)
        return matches

    def getDefaultMarkers(self) -> list:
        """Returns list of planar coordinates of the default markers for the saw in 
        resting position."""
        uvs1 = [[761, 1231], [1061, 1354], [1343, 1360], [2162, 942], [1642, 618], [984, 15]]
        uvs2 = [[1525, 668], [2102, 1028], [1054, 1503], [753, 1476], [627, 1341], [826, 29]]
        return uvs1, uvs2

    def getImage(self, img_path):
        """Returns the image at the filepath as an array_like object."""
        img = cv.imread(img_path)
        return img
    
    def makeGrayscale(self, img):
        """Removes color in the image leaving only shades of black, grey, and white."""
        out = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        return out
    
    def blurImage(self, img, radius=15):
        """Performs Gaussian blurring on the image with given radius."""
        blurred = cv.GaussianBlur(img, (radius, radius), 0)
        return blurred
    
    def findBrightestPoint(self, reduced_img, point: tuple=None, dx: float=0.05, dy: float=None) -> tuple:
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
            reduced_img, limits = self.getRegionFromImage(reduced_img, point, dx, dy)
        (minVal, maxVal, minLoc, maxLoc) = cv.minMaxLoc(reduced_img)
        point = (limits[0] + maxLoc[0], 
                 limits[2] + maxLoc[1])
        # cv.drawMarker(reduced_img, (maxLoc[0], maxLoc[1]), (0, 0, 255), cv.MARKER_DIAMOND, 20, 2, 1)
        # cv.imshow("Brightest Point", reduced_img)
        # cv.waitKey(0)
        # cv.destroyAllWindows()
        return point
        
    def getRegionFromImage(self, img, point: tuple, dx: float=0.05, dy: float=None):
        """Clips the given image to a box region centered at the `point` with width dx
        and height dy.
        
        Returns
        -------
            array_like: 2D subset of `img`
            tuple: 4x1 tuple with coordinates of the region as `(min_x, max_x, min_y, max_y)`,
                so that the outputted image can be calculated as `img[min_y : max_y, min_x : max_x]`
        """
        width = min([len(i) for i in img])
        height = len(img)
        if dy is None: 
            dy = dx
        dx = np.round(dx * width)
        dy = np.round(dy * height)

        min_x = np.rint(max(0, point[0] - dx)).astype(int)
        max_x = np.rint(min(width, point[0] + dx)).astype(int)
        min_y = np.rint(max(0, point[1] - dy)).astype(int)
        max_y = np.rint(min(height, point[1] + dy)).astype(int)
        img = np.array(img)
        clipped_img = img[min_y : max_y, min_x : max_x]
        return clipped_img, (min_x, max_x, min_y, max_y)

    def displayMarkers(self, marker_coords: list, img_path: str=None, img_frame=None):
        """Displays the markers on the given image.
        
        Wrapper for `OpenCV.drawMarker` function. Either`img_path` and `img_frame` 
        must be passed to the function.

        Parameters
        ----------
        marker_coords: list
            List of (2x1) array_like pixel coordinates for each marker.
        img_path: str, Optional
            Filepath for the image to display the markers on.
        img_frame: list, Optional
            The read frame in the form of `cv.InputOutputArray`
        """
        if img_path is not None:
            frame = cv.imread(img_path, 1)
        elif img_frame is None:
            raise(Exception("No image path or IOarray was passed to display markers"))
        else: frame = img_frame

        for m in marker_coords:
            cv.drawMarker(frame, m[:2], color=(0, 0, 255), 
                          markerType=cv.MARKER_CROSS, markerSize=20, thickness=2)
        cv.imshow("Marker Positioning", frame)
        cv.waitKey(0)
        cv.destroyAllWindows()

    def writeSequence2DB(self, labels: list=['x', 'y', 'z']):
        """Writes a sequence of marker coordinates to the Database."""
        mocapseq_table = self.db.name.MOCAPSEQ_TBL_NAME
        seq_pk = self.db.getMaxPrimaryKey(mocapseq_table) + 1

        self.db.addEntry(mocapseq_table, [seq_pk], [self.db.name.MOCAPSEQ_TBL_COLS[0]])

        for marker in self.markers.values():
            mkr_pk = marker['pk']
            coords = marker['coords']
            for i, coord_label in enumerate(labels):
                value = coords[i]
                self.db.addEntry(self.db.name.MOCAP_TBL_NAME,
                        [coord_label, value, seq_pk, mkr_pk], 
                        self.db.name.MOCAP_TBL_COLS[1:])

    def writeMarkers2DB(self, marker_labels: list):
        """Adds the given markers to the Database and updates the class memory."""
        table_name = self.db.name.MOCAPMKR_TBL_NAME
        pk = self.db.getMaxPrimaryKey(table_name) + 1
        db_markers = self.db.getEntries(table_name)
        extant_labels = [e[1] for e in db_markers]
        for l in marker_labels:
            if l not in extant_labels:
                self.db.addEntry(table_name, [pk, l], self.db.name.MOCAPMKR_TBL_COLS)
                pk += 1
                self.markers[l] = dict(pk=pk, coords=[0]*3)
            else:
                db_pk = db_markers[extant_labels.index(l)][0]
                self.markers[l] = dict(pk=db_pk, coords=[0]*3)





    

