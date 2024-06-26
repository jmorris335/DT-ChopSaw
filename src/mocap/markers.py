"""
| File: marker_detection.py 
| Info: Presents class for working with aruco markers using openCV methods
| Author: John Morris, jhmrrs@clemson.edu  
| Organization: Product Lifecycle Management Center at Clemson University, plmcenter@clemson.edu  
| Permission: Copyright (C) 2023, John Morris. All rights reserved. Should not be reproduced, edited, sourced, or utilized without written permission from the author or organization

| Version History:
| - 0.0, 5 Jun 2024: Initialized
"""
import numpy as np
import cv2 as cv
import yaml

class Marker:
    def __init__(self, name: str, id: int, db_id: int=None):
        """Container class for marker points.
        
        Parameters
        ----------
        name : str
            The label of the marker
        id : int
            The dictionary id for the marker
        db_id : int
            The primary key for the marker in the database
        """
        self.name = name
        self.id = id
        self.db_id = db_id
        self.centers = list()
        self.T_w = [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]

    def point(self) -> tuple:
        """Gets the latest value for the center of the marker."""
        return self.centers[-1]

    def triangulateCenter(self, corners, proj_mtrxs, cam_mtrxs, dist_coefs) -> tuple:
        """Calculates the center of the marker from the corners.

        Parameters
        ----------
        corners : list
            A 4x2x2 array of 1x2 tuples of floats representing the coordinate 
            points for the marker's corners in a camera plane. This is of the form 
            `[[(xl_1, yl_1), (xr_1, yr_1)], [(xl_2, yl_2), (xr_2, yr_2)], ...]`.
        proj_mtrx : list
            A list of length two of the 4x4(?) projection matrices.
        cam_mtrx : list
            A list of length two of the 3x3 intrinsic camera matrices.
        dist_coefs : list
            A list of length two of the distortion coefficients vectors.
        """
        corners3D = self.planar2global(*corners, proj_mtrxs, cam_mtrxs, dist_coefs)
        center = np.mean(corners3D, 0)
        self.centers.append(center)
        return center

    def planar2global(self, pts1, pts2, proj_mtrxs: list, cam_mtrxs: list, dist_coefs: list) -> tuple:
        """Converts each matching point in `pts1` and `pts2` to a triangulated 
        3D point.
        
        Parameters
        ----------
        pts1, pts2 : array_like
            A (N x 2) array of points represnting the same 3D point as seen from 
            two stereographic cameras.
        cam_mtrx : list
            A list of length two of the 3x3 intrinsic camera matrices.
        proj_mtrx : list
            A list of length two of the 4x4(?) projection matrices.
        dist_coefs : list
            A list of length two of the distortion coefficients vectors.

        Returns
        -------
        - N x 3 array where each row `i` is a tuple representing the (x, y, z) 
            coordinates of the point referenced by `pts1[i]` and `pts2[i]`.
        """
        pts = [np.array(p, 'float32') for p in [pts1, pts2]]
        pts_u = [cv.undistortPoints(pts[i], cam_mtrxs[i], dist_coefs[i], None, 
                                    cam_mtrxs[i]) for i in range(2)]
        
        pts_4D = cv.triangulatePoints(proj_mtrxs[0], proj_mtrxs[1], *pts_u)
        pts_4D = self.T_w @ pts_4D
        pts_3D = (pts_4D[:3, :]/pts_4D[3, :]).T
        return tuple(pts_3D[0])

class Aruco:
    """Class for interacting with Aruco markers, largely employing OpenCV methods."""
    def __init__(self, marker_info_filepath="src/mocap/marker_info.yaml"):
        """Creates an object to handle ArUco marker functionality.
        
        Parameters
        ----------
        marker_info_filepath : str, default="src/mocap/marker_info.yaml"
        """
        with open(marker_info_filepath, 'r') as file:
            marker_data, param_data = yaml.safe_load_all(file)
        ARUCO_DICT_IDX = getattr(cv.aruco, marker_data['DICTIONARY'])
        self.PIXEL_SIZE = marker_data['PIXEL_SIZE']
        self.ARUCO_DICT = cv.aruco.getPredefinedDictionary(ARUCO_DICT_IDX)
        self.markers = dict()
        for m in marker_data['markers']:
            self.markers[m['name']] = Marker(**m)
        self.param_names = param_data['parameters']
        self.aruco_detector = cv.aruco.ArucoDetector(self.ARUCO_DICT)

    def generateMarkers(self, show: bool=False) -> list:
        """Generates the set of ArUco markers."""
        for m in self.markers.values():
            am = cv.aruco.generateImageMarker(self.ARUCO_DICT, m.id, self.PIXEL_SIZE)
            filepath = f"src/mocap/demo/ArUco/{m.name}_marker{m.id}.png"
            cv.imwrite(filepath, am)
            cv.imshow("Marker ID " + str(m.id), am)
            if show:
                cv.waitKey(1000)   

    def findMarkerCenters(self, frames, proj_mtrxs, cam_mtrxs, dist_coefs):
        """Calculates the centers for all markers found in the given frames."""
        corners1, ids1 = self.detectMarkers(frames[0])
        corners2, ids2 = self.detectMarkers(frames[1])
        self.calculateCenters([corners1, corners2], [ids1, ids2], proj_mtrxs, cam_mtrxs, dist_coefs)

    def detectMarkers(self, frame: np.ndarray, show: bool=False):
        """Detects all aruco markers in the frame and returns the 2D positions of
        their four corners and the marker IDs.
        """
        #TODO: MAKE SURE IMAGE IS UNDISTORTED!
        corners, ids, _ = self.aruco_detector.detectMarkers(frame)
        if show:
            self.showDetectedMarkers(frame, corners, ids)
        return corners, ids
    
    def calculateCenters(self, corners, ids, proj_mtrxs, cam_mtrxs, dist_coefs):
        """Calculates the centers of the markers at the given IDs and corners."""
        for id in ids:
            m = self.getMarkerByID(id)
            m.triangulateCenter(self, corners, proj_mtrxs, cam_mtrxs, dist_coefs) 

    def getMarkerByID(self, id) -> Marker:
        """Returns the first marker in `self.markers` with the given ID."""
        for m in self.markers.values():
            if m.id == id:
                return m
        return None
    
    def showDetectedMarkers(self, frame, corners, ids):
        debug_frame = frame.copy()
        cv.aruco.drawDetectedMarkers(debug_frame, corners, ids)
        cv.imshow("Detected ArUco Markers", debug_frame)
        cv.waitKey(-1)

if __name__ == "__main__":
    a = Aruco()
    # a.generateMarkers(show=False)
    frame = cv.imread("src/mocap/dev-mocap/ArUco Test.jpg")
    a.detectMarkers(frame, show=True)