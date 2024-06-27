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
    def __init__(self, name: str, id: int, moffset: tuple=(0,0,0), db_id: int=None,
                 center: float=None):
        """Container class for marker points.
        
        Parameters
        ----------
        name : str
            The label of the marker
        id : int
            The dictionary id for the marker
        moffset : tuple, default=(0,0,0)
            The offset for the marker in model coordiantes from the ideal 
            position of the joint the marker represents when the joint is at 
            rest. These ideal positions are:
                base: miter center of rotation, flush with table
                miter: miter center of rotation, flush with table
                bevel: bevel center of rotation, flush with table
                slider: center and top of beveling stem, center of slider arm
                crash: crash arm center of rotation, center of slider arm
        db_id : int, Optional
            The primary key for the marker in the database
        center : float, Optional
            A length-3 tuple in model coordinates representing the center of the
            marker
        """
        self.name = name
        self.id = id
        self.moffset = moffset
        self.db_id = db_id
        self.center = list()
        self.T_w = [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]

    def triangulateCenter(self, corners, proj_mtrxs, cam_mtrxs, dist_coefs, 
                          world2model=None) -> tuple:
        """Calculates the center of the marker from the corners, which is returned
        in world coordinates unless `world2model` is passed."

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
        world2model : array_like, Optional
            A 4x4 matrix detailing the transformation from world coordinates to 
            model coordinates, such that `world @ world2model = model`
        """
        corners3D = self.planar2global(*corners, proj_mtrxs, cam_mtrxs, dist_coefs)
        center = np.mean(corners3D, 0)
        if world2model is not None:
            center = np.append(center, 0) @ world2model
        self.center = center
        return center

    def planar2global(self, pts1, pts2, proj_mtrxs: list, cam_mtrxs: list, dist_coefs: list) -> tuple:
        """Converts each matching point in `pts1` and `pts2` to a triangulated 
        3D point.
        
        Parameters
        ----------
        pts1, pts2 : array_like
            A (N x 2) array of points representing the same 3D point as seen from 
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
    
    def calcModelCSYS(self, BL: tuple, BR: tuple, TL: tuple, 
                      TR: tuple=None)-> np.ndarray:
        """ Calculates the Rotation matrix that aligns the world coordinate frame 
        with the model coordinate frame. 
        
        Parameters
        ----------
        BL, BR, TL, TR: tuple
            World CSYS coordinates for the corners of a marker that is aligned with 
            the Model CSYS (such as the base), entered as length-3 tuples and given
            as: Bottom-Left, Bottom-Right, Top-Left, with Top-Right as an unused 
            parameter.
        """
        BL, BR, TL = np.array((BL, BR, TL))
        x = BR - BL
        y = TL - BL
        z = np.cross(x, y)
        x, y, z = [a / np.linalg.norm(a) for a in (x, y, z)]
        M = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]) #Model CSYS
        R = np.linalg.inv([x, y, z]) @ M
        T = np.row_stack((np.column_stack((R, np.zeros((3,1))), [*self.moffset, 1])))
        return T

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
        self.world2model = np.eye(4)

    def generateMarkers(self, show: bool=False) -> list:
        """Generates the set of ArUco markers. Returns a list of filepaths for
        each marker."""
        filepaths = list()
        for m in self.markers.values():
            am = cv.aruco.generateImageMarker(self.ARUCO_DICT, m.id, self.PIXEL_SIZE)
            filepath = f"src/mocap/media/ArUco/{m.name}_marker{m.id}.png"
            cv.imwrite(filepath, am)
            filepaths.append(filepath)
            if show:
                cv.imshow("Marker ID " + str(m.id), am)
                cv.waitKey(1000)
        return filepaths

    def findMarkerCenters(self, frames, proj_mtrxs, cam_mtrxs, dist_coefs, show=False):
        """Calculates the centers for all markers found in the given frames."""
        corners1, ids1 = self.detectMarkers(frames[0], show)
        corners2, ids2 = self.detectMarkers(frames[1], show)
        if ids1 is None or ids2 is None: return None #A camera found no markers
        ids1, ids2 = [[i[0] for i in ids] for ids in (ids1, ids2)]
        corners1, corners2 = [[c[0].tolist() for c in corners] for corners in (corners1, corners2)]
        corners, ids = self.coordinateFoundMarkers(corners1, corners2, ids1, ids2)
        self.calculateCenters(corners, ids, proj_mtrxs, cam_mtrxs, dist_coefs)

    def detectMarkers(self, frame: np.ndarray, show: bool=False):
        """Detects all aruco markers in the frame and returns the 2D positions of
        their four corners and the marker IDs.
        """
        #TODO: MAKE SURE IMAGE IS UNDISTORTED!
        corners, ids, _ = self.aruco_detector.detectMarkers(frame)
        if show:
            self.showDetectedMarkers(frame, corners, ids)
        return corners, ids
    
    def coordinateFoundMarkers(self, corners1: list, corners2, ids1: list, ids2: list):
        """Updates the corners found on each marker in the list that was seen by 
        both cameras. Corner inputs are nx4x2 array_like. Ids is a length-n list."""
        common_ids = list(set(ids1).intersection(ids2))
        common_corners = list()
        for id in common_ids:
            c1 = corners1[ids1.index(id)]
            c2 = corners2[ids2.index(id)]
            common_corners.append([c1, c2])
        return common_corners, common_ids 
    
    def calculateCenters(self, corners, ids: list, proj_mtrxs, cam_mtrxs, dist_coefs):
        """Calculates the centers of the markers at the given IDs and corners, 
        where corners is nx2x4x2 and ids is length n."""
        for id in ids:
            m = self.getMarkerByID(id)
            if m.name == 'base':
                base_corners = corners[ids.index(id)]
                world_corners = m.planar2global(*base_corners, proj_mtrxs, cam_mtrxs, dist_coefs)
                self.world2model = m.calcModelCSYS(*world_corners)
            m.triangulateCenter(self, corners, proj_mtrxs, cam_mtrxs, dist_coefs, 
                                self.world2model)       

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

def printMarkers(filepath='src/mocap/media/ArUco/'):
    """A helper function for generating a printable doc of the ArUco markers."""
    import matplotlib.pyplot as plt
    aruco = Aruco()
    paths = aruco.generateMarkers()
    nrows = len(paths) // 4 + 1
    ncols = 4
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*2.3, nrows*2.5))
    for ax in fig.axes:
        ax.axis("off")
    for i,p in enumerate(paths):
        index = (i // ncols, i % ncols)
        img = cv.imread(p)
        axs[index].imshow(img)
        last_backslash = p.rindex('/') + 1
        file_name = p[p.rindex('/') + 1 : p.index('.png')]
        marker_name, _, marker_id = file_name.partition('_')
        axs[index].set_title(marker_name)
        axs[index].text(0.5, -0.1, marker_id, ha="center", transform=axs[index].transAxes)
    plt.savefig(filepath + 'printable.png')
    plt.show()

if __name__ == "__main__":
    # a = Aruco()
    # # a.generateMarkers(show=False)
    # frame = cv.imread("src/mocap/dev-mocap/ArUco Test.jpg")
    # a.detectMarkers(frame, show=True)
    printMarkers()