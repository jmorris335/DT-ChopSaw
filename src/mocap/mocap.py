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
import logging as log
import time
import threading

from src.mocap.calibration import calibrateCameraIntrinsic, stereoCalibration
from src.mocap.camera_interface import VideoInterface
from src.auxiliary.geometry import pointDistance
from src.db.actor import DBActor

log.basicConfig(level=log.DEBUG)

class Mocap():
    """Performs linear transformations of planar points into 3D space."""
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

        def planar2global(self, projection_mtrxs: list) -> tuple:
            """Triangulates the 3D coordinates from the two 2D coordinates."""
            points2D = np.array(self.planar_coords, 'float').T
            points4D = cv.triangulatePoints(*projection_mtrxs[:2], *points2D[:2])
            points3D = points4D[:3] / points4D[-1]
            self.global_coord = tuple(points3D.T[0])
            return points4D[:3]

            # self.global_coord = self.DLT(*projection_mtrxs, *self.planar_coords)
        
        @staticmethod
        def DLT(P1, P2, point1, point2):
            A = [point1[1]*P1[2,:] - P1[1,:],
                P1[0,:] - point1[0]*P1[2,:],
                point2[1]*P2[2,:] - P2[1,:],
                P2[0,:] - point2[0]*P2[2,:]
                ]
            A = np.array(A).reshape((4,4))
        
            B = A.transpose() @ A
            U, s, Vh = np.linalg.svd(B, full_matrices = False)

            global_coord = Vh[3,0:3]/Vh[3,3]
            return global_coord

    def __init__(self, camera1, camera2, projection_mtrxs: list=None, calib_frames=None):
        """
        Parameters
        ----------
        camera1 : str | int
            Either a filepath to a video stream or the camera ID.
        camera2 : str | int
            Same as `camera1` for the second camera.
        projection_mtrcs : list, Optional
            A list of projection matrices (3x4)
       
        """
        self.cameras = [
            cv.VideoCapture(camera1),
            cv.VideoCapture(camera2)
        ]

        # Make sure camera is calibrated
        if projection_mtrxs is None:
            self.is_calibrated = False
            if calib_frames is None:
                calib_frames = self.getCalibrationFrames()
            self.calibrate(calib_frames)
        else:
            self.projection_mtrxs = projection_mtrxs
            self.is_calibrated = True
            
        # Initialize Database
        self.db = DBActor()
        self.markers = self.getDefaultMarkers()
        self.setupMarkerDB(['base_1', 'base_2', 'miter_1', 'bevel_1', 'arm_1', 'arm_2'])

    def __del__(self):
        for cam in self.cameras:
            cam.release()

    def start(self):
        """Begins motion capture process."""
        frame_counter = 0
        while all([cam.isOpened() for cam in self.cameras]):
            log.debug(f"Processing frame {frame_counter}")
            self.mocapProcess()
            frame_counter += 1

    def getMarker(self, label: str) -> Marker:
        """Returns the marker at the given label."""
        for m in self.markers:
            if m.label == label:
                return m
        raise Exception(f'Label "{label}" not found in marker list.')
    
    def getCamIdx(self, cam) -> int:
        """Returns the index of the camera for the class."""
        return self.cameras.index(cam)

    def getCalibrationFrames(camera_ids: list=[0, 1]):
        """Opens cameras and collects calibration images."""
        dir_path = "src/mocap/media"
        num_frames = 12
        vi = VideoInterface([0, 1], dir_path=dir_path)
        vi.getCalibrationFrames()
        calib_frames = [
            [f"{dir_path}/calib/cam01_{i}.png" for i in range(num_frames)],
            [f"{dir_path}/calib/cam02_{i}.png" for i in range(num_frames)]
        ]
        return calib_frames

    def calibrate(self, calib_img_paths: list, 
                  num_rows: int=4, num_cols: int=7, square_size: float=2.5):
        """Calibrates the class for according to a calibration object for use in arbitrary
        coordination.
        
        Paramters
        ---------
        """
        cam_mtrxs, cam_dist_coeffs = list(), list()
        for imgs in calib_img_paths:
            calib_consts = calibrateCameraIntrinsic(imgs, num_rows, num_cols, 
                                                    square_size, show_images=False)
            
            log.debug(f"Camera {calib_img_paths.index(imgs)} RMSE: {calib_consts['rmse']}")
            cam_mtrxs.append(calib_consts['camera_matrix'])
            cam_dist_coeffs.append(calib_consts['dist_coeffs'])
    
        rmse, R, T = stereoCalibration(*calib_img_paths[:2], *cam_mtrxs, *cam_dist_coeffs, 
                      num_rows, num_cols, square_size, show_images=False)
        log.debug(f"Stereo calibration RMSE: {rmse}")

        self.projection_mtrxs = list()
        RT1 = np.concatenate([np.eye(3), [[0],[0],[0]]], axis = -1)
        P1 = R @ RT1 #projection matrix for C1
        self.projection_mtrxs.append(P1)
    
        RT2 = np.concatenate([R, T], axis = -1)
        P2 = R @ RT2 #projection matrix for C2
        self.projection_mtrxs.append(P2)

        self.is_calibrated = True
        log.info("Calibrated cameras")
    
    def getDefaultMarkers(self) -> list:
        """Returns list of planar coordinates of the default markers for the saw in 
        resting position."""
        labels = ['base_1', 'base_2', 'miter_1', 'bevel_1', 'arm_1', 'arm_2']
        coord = [[[761, 1231], [1061, 1354], [1343, 1360], [2162, 942], [1642, 618], [984, 15]],
                 [[1525, 668], [2102, 1028], [1054, 1503], [753, 1476], [627, 1341], [826, 29]]]
        
        markers = list()
        for i in range(len(labels)):
            markers.append(self.Marker(labels[i], [coord[0][i], coord[1][i]], (0,)*3))
        return markers

    def mocapProcess(self):
        """Conducts a basic motion capture routine on the next available frame"""
        new_coords = list()
        for cam in self.cameras:
            img = self.getImage(cam)
            if img is None: return
            new_coords.append(self.getCoordsInImage(img, self.getCamIdx(cam)))
            self.writeImageToWebapp(img, cam, new_coords[self.getCamIdx(cam)])
        self.assignCoordsToMarkers(new_coords)
        for m in self.markers:
            m.planar2global(self.projection_mtrxs)
        seq = self.compileSequence()
        self.writeSequence2DB(seq)
        # db_thread = threading.Thread(target=self.writeSequence2DB, args=[seq], daemon=True)
        # db_thread.start()

    def writeImageToWebapp(self, img, camera, coords: list=None):
        """Writes the image to the assets folder of the webapp."""
        if coords is not None:
            img = self.drawCoordinates(coords, img_frame=img)
        img = cv.flip(img, 1)
        cam_idx = self.getCamIdx(camera)
        cv.imwrite(f"src/gui/assets/live_feed_cam{cam_idx}.png", img)

    def getCoordsInImage(self, img, cam_idx: int=0) -> list:
        """Returns coordinates for the brightest points in the image."""
        grayscale = self.makeGrayscale(img)
        blurred = self.blurImage(grayscale)
        new_coords = list()
        old_coords = [m.planar_coords[cam_idx] for m in self.markers]
        for old_cord in old_coords:
            new_coords.append(self.findBrightestPoint(blurred, old_cord))
        return new_coords
  
    def assignCoordsToMarkers(self, new_coords: list):
        """Assigns new coordinates to markers in class."""
        #Find which marker is closest to a skeleton point
        for cam in range(len(new_coords)):
            unmatched_markers = [m for m in self.markers]
            for pt in new_coords[cam]:
                d = [pointDistance(pt, m.planar_coords[cam]) for m in unmatched_markers]
                idx = d.index(min(d))
                unmatched_markers[idx].planar_coords[cam] = pt
                del(unmatched_markers[idx])

    ## Image Processing
    def getImage(self, camera: cv.VideoCapture):
        """Returns the image at the filepath as an array_like object."""
        img = camera.read()
        if not img[0]:
            camera.release()
        return img[1]
    
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
    
    def drawCoordinates(self, coords: list, img_path: str=None, img_frame=None):
        """Draws the markers on the given image.
        
        Wrapper for `OpenCV.drawMarker` function. Either`img_path` and `img_frame` 
        must be passed to the function.

        Parameters
        ----------
        coords: list
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

        for coord in coords:
            cv.drawMarker(frame, coord[:2], color=(0, 0, 255), markerSize=20, markerType=cv.MARKER_CROSS, thickness=2)
        return frame

    def displayCoordinates(self, coords: list, img_path: str=None, img_frame=None):
        """Displays the markers on the given image.
        
        Wrapper for `OpenCV.drawMarker` function. Either`img_path` and `img_frame` 
        must be passed to the function.

        Parameters
        ----------
        coords: list
            List of (2x1) array_like pixel coordinates for each marker.
        img_path: str, Optional
            Filepath for the image to display the markers on.
        img_frame: list, Optional
            The read frame in the form of `cv.InputOutputArray`
        """
        frame = self.drawCoordinates(coords, img_path, img_frame)
        cv.imshow("Marker Positioning", frame)
        cv.waitKey(0)
        cv.destroyAllWindows()

    ## DB Methods
    def setupMarkerDB(self, marker_labels: list):
        """Adds the given markers to the Database and updates the class variable."""
        table_name = self.db.name.MOCAPMKR_TBL_NAME
        pk = self.db.getMaxPrimaryKey(table_name) + 1
        db_markers = self.db.getEntries(table_name)
        extant_labels = [e[1] for e in db_markers]
        for l in marker_labels:
            if l not in extant_labels:
                self.db.addEntry(table_name, [pk, l], self.db.name.MOCAPMKR_TBL_COLS)
                pk += 1
                self.getMarker(l).db_key = pk
            else:
                db_pk = db_markers[extant_labels.index(l)][0]
                self.getMarker(l).db_key = db_pk

    def compileSequence(self, labels: list=['x', 'y', 'z']) -> list:
        """Compiles a sequence based on current markers to write to the database.
        
        A sequence is a list of the global coordinates for each marker at a 
        specific timestamp. This function the sequence as a list of dictionaries,
        where each dict links the variable name, the coordinate value, and the 
        db key for the marker.
        """
        seq = list()
        for marker in self.markers:
            coords = marker.global_coord

            for i, label in enumerate(labels):
                value = float(coords[i])
                seq.append({'label' : label, 
                            'value' : value, 
                            'mkr_key' : marker.db_key})
        return seq

    def writeSequence2DB(self, seqs: list):
        """Writes a sequence of marker coordinates to the Database."""
        mocapseq_table = self.db.name.MOCAPSEQ_TBL_NAME
        seq_pk = self.db.getMaxPrimaryKey(mocapseq_table) + 1

        self.db.addEntry(mocapseq_table, [seq_pk], [self.db.name.MOCAPSEQ_TBL_COLS[0]])

        tbl_name = self.db.name.MOCAP_TBL_NAME
        col_names = self.db.name.MOCAP_TBL_COLS[1:]
        for seq in seqs:
            values = [seq['label'], seq['value'], seq_pk, seq['mkr_key']]
            self.db.addEntry(tbl_name, values, col_names)
 