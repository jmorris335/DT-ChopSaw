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
    __slots__ = ("label", "planar_coords", "global_coord", "db_key")
    def __init__(self, label: str, planar_coords: list, global_coord: tuple, db_key: int=None):
        """Container class for marker points.
        
        Parameters
        ----------
        label : str
            The label for the marker
        planar_coords : list
            A 2x2 array of 1x2 tuples of floats representing the coordinate 
            points for the marker in the camera plane. This is of the form 
            `[(x1, y1), (x2, y2)]`.
        global_coord : tuple
            A 3x1 tuple representing the marker's coordinates in Euclidean 
            geometry
        db_key : int
            The database key for the marker
        """
        self.label = label
        self.db_key = db_key
        self.planar_coords = planar_coords
        self.global_coord = global_coord

    def planar2global(self, camera_mtrx: list, projection_mtrxs: list, dist_coeffs: list) -> tuple:
        # points1 is a (N, 1, 2) float32
        points1 = np.array(self.planar_coords[0], 'float32')
        points2 = np.array(self.planar_coords[1], 'float32')

        points1u = cv.undistortPoints(points1, camera_mtrx[0], dist_coeffs[0], None, camera_mtrx[0])
        points2u = cv.undistortPoints(points2, camera_mtrx[1], dist_coeffs[1], None, camera_mtrx[1])

        points4d = cv.triangulatePoints(projection_mtrxs[0], projection_mtrxs[1], points1u, points2u)
        T_w = [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]
        points4d = T_w @ points4d
        points3d = (points4d[:3, :]/points4d[3, :]).T
        self.global_coord = tuple(points3d[0])

class Aruco:
    """Class for interacting with Aruco markers, largely employing OpenCV methods."""
    class ArucoMarker:
        """Generic OO class for structuring a basic Aruco Marker."""
        __slots__ = ("name", "id")
        def __init__(self, name: str, id: int):
            self.name = name
            self.id = id

    def __init__(self, marker_info_filepath="src/mocap/marker_info.yaml"):
        """Creates an object to handle ArUco marker functionality.
        
        Parameters
        ----------
        marker_info_filepath : str, default="src/mocap/marker_info.yaml"
        """
        with open(marker_info_filepath, 'r') as file:
            data = yaml.safe_load(file)

        ARUCO_DICT_IDX = getattr(cv.aruco, data['DICTIONARY'])
        self.PIXEL_SIZE = data['PIXEL_SIZE']
        self.ARUCO_DICT = cv.aruco.getPredefinedDictionary(ARUCO_DICT_IDX)
        self.markers = [self.ArucoMarker(**m) for m in data['markers']]
        self.aruco_detector = cv.aruco.ArucoDetector(self.ARUCO_DICT)

    def generateMarkers(self, show: bool=False) -> list:
        """Generates the set of ArUco markers."""
        for m in self.markers:
            am = cv.aruco.generateImageMarker(self.ARUCO_DICT, m.id, self.PIXEL_SIZE)
            filepath = f"src/mocap/demo/ArUco/{m.name}_marker{m.id}.png"
            cv.imwrite(filepath, am)
            cv.imshow("Marker ID " + str(m.id), am)
            if show:
                cv.waitKey(1000)            

    def detectMarkers(self, frame: np.ndarray, show: bool=False) -> dict:
        """Detects all aruco markers in the frame and returns the 2D positions of
        their four corners and the marker IDs.
        """
        #TODO: MAKE SURE IMAGE IS UNDISTORTED!
        corners, ids, _ = self.aruco_detector.detectMarkers(frame)
        if show:
            debug_frame = frame.copy()
            cv.aruco.drawDetectedMarkers(debug_frame, corners, ids)
            cv.imshow("Detected ArUco Markers", debug_frame)
            cv.waitKey(-1)
        return corners, ids

if __name__ == "__main__":
    a = Aruco()
    a.generateMarkers(show=False)
    # frame = cv.imread("src/mocap/dev-mocap/ArUco Test.jpg")
    # a.detectMarkers(frame, show=True)