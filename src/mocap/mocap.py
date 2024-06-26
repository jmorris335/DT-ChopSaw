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

from src.mocap.calibration import calibCamIntrinsic, stereoCalibration
from src.mocap.camera_interface import VideoInterface
from src.mocap.markers import Marker, Aruco
from src.auxiliary.geometry import pointDistance
from src.auxiliary.transform import Transform
from src.db.actor import DBActor

log.basicConfig(level=log.DEBUG)

__slots__ = ("label", "db_id", "values")
class Param:
    """Container for a generic parameter to be inserted into the DB."""
    def __init__(self, label: str, db_id: int, calculated_value: float=0.0):
        self.label = label
        self.db_id = db_id       
        self.value = calculated_value
        self.recorded_values = list()

    def push(self, value: float, max_size: int=9):
        """Adds the value to the object and slices the array if too large."""
        self.recorded_values.append(value)
        self.recorded_values[-max_size:]

    def median(self, size: int=None) -> float:
        """Returns the median of the parameter values (scooping from the end)."""
        if size is None:
            return np.median(self.recorded_values)
        return np.median(self.recorded_values[-size:])
    
    def set(self, value: float, max_size: int=None, use_median: bool=False, median_size: int=None):
        """Adds the new value to the object and sets the current value as either
        `value` or the median of the `recorded_values`, depending on `use_median`.
        """
        if max_size is None:
            self.push(value)
        else: self.push(value, max_size)
        if use_median:
            self.value = self.recorded_values[-1]
        else:
            self.value = self.median(median_size)

    def sample(self, size: int=3) -> list:
        """Returns the last several elements of the parameter values."""
        return self.recorded_values[-size:]
    

def startMocap(camera1, camera2, projection_mtrxs: list=None, calib_frames=None):
    """Caller for the motion capture process"""
    cameras = openCameras([camera1, camera2])
    if projection_mtrxs is None:
        if calib_frames is None:
            calib_frames = getCalibrationFrames()
        proj_mtrxs, cam_mtrxs, dist_coefs = calibrate(calib_frames)

    doMocap(cameras, proj_mtrxs, cam_mtrxs, dist_coefs)
    releaseCameras(cameras)

def openCameras(cameras: list):
    """Opens the cameras."""
    return [cv.VideoCapture(cam) for cam in cameras]

def getCalibrationFrames(camera_ids: list=[0, 1]) -> list:
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

def calibrate(self, calib_imgs: list, n_rows: int=4, n_cols: int=7, 
                  square_size: float=2.5):
    """Retrieves the calibration constants from a set of given images.
    
    Parameters
    ----------
    calib_imgs : list
        list of size 2, where each entry is a list of images taken from a 
        single camera. Each image is of a checkerboard captured by each 
        camera at a single instant (so that each image is synchronized). See
        `camera_interface.py` for assistance in taking the images.
    n_rows: int=4
        The number of rows of intersections on the checkerboard
    n_cols: int=7
        The number of columns of intersections on the checkerboard
    square_size: float=2.5
        The dimension of a single square on the checkerboard, in cm

    Returns
    -------
    list : length-2 list of projection matrices
    list : length-2 list of lists of distortion coefficients
    list : length-2 list of 3x3 camera intrinsic matrices
    """
    calib = [calibCamIntrinsic(calib_imgs[i], n_rows, n_cols, square_size, 
                                False) for i in range(len(calib_imgs))]
    mtx = [calib[i]['camera_matrix'] for i in range(len(calib))]
    dist = [calib[i]['dist_coeffs'] for i in range(len(calib))]
    rmse, R, T = stereoCalibration(*calib_imgs, *mtx, *dist, 4, 7, 2.5, False)
    projMat1 = mtx[0] @ cv.hconcat([np.eye(3), np.zeros((3,1))]) # Cam1 is the origin
    projMat2 = mtx[1] @ cv.hconcat([R, T]) # R, T from stereoCalibrate

    return [projMat1, projMat2], dist, mtx

def releaseCameras(cameras: list):
    """Releases the cameras once the sequencing is complete."""
    for cam in cameras:
        cam.release()

def doMocap(cameras: list, proj_mtrxs: list, cam_mtrxs: list, dist_coefs: list):
    """Caller for the main Mocap process."""
    aruco = Aruco()
    db = DBActor()
    params = setupMocapDB(params, db)
    frame_counter = 0
    while all([cam.isOpened() for cam in cameras]):
        log.debug(f"Processing frame {frame_counter}")
        updateMarkers(cameras, proj_mtrxs, cam_mtrxs, dist_coefs, aruco)
        frame_counter += 1
        params = updateParamsFromMarkers(params, aruco)
        sendParamsToDB(params)

def updateMarkers(cameras: list, proj_mtrxs: list, cam_mtrxs: list, 
                  dist_coefs: list, aruco: Aruco) -> dict:
    """Updates 3D coordinates for each marker in the list found by both cameras."""
    imgs = [getImage(cam) for cam in cameras]
    if not all(imgs): 
        return #Check for invalid frames
    aruco.findMarkerCenters(imgs, proj_mtrxs, cam_mtrxs, dist_coefs)
    
def getImage(camera):
    """Returns the next frame from the given camera."""
    img = camera.read()
    if not img[0]:
        camera.release()
    return img[1]

def updateParamsFromMarkers(aruco: Aruco, params: dict) -> dict:
    """Returns a dict of `Param` objects calculated from the aruco markers."""
    markers = aruco.markers
    updateOrigin(markers, params)
    updateBumpOffset(markers, params)
    updateMiterAngle(markers, params)

def updateOrigin(markers: dict, params: dict):
    """Calculates the origin (x, y, z) as the median of the found coordinates."""
    labels = ['origin_x', 'origin_y', 'origin_z']
    origin_1 = markers['origin_1'].point()
    origin_2 = markers['origin_2'].point()
    for i in range(len(labels)):
        value_avg = np.mean([origin_1[i], origin_2[i]])
        params[labels[i]].set(value_avg, use_median=True, median_size=4)

def updateBumpOffset(markers: dict, params: dict):
    """Calculates the bump offset (x, y, z) as the median of the found coordinates."""
    bo_labels = ['bump_offset_x', 'bump_offset_y', 'bump_offset_z']
    origin_labels = ['origin_x', 'origin_y', 'origin_z']
    base_1 = markers['base_1'].point()
    base_2 = markers['base_2'].point()
    for i in range(len(bo_labels)):
        base_avg = np.mean([base_1[i], base_2[i]])
        bo = params[origin_labels[i]] - base_avg
        params[bo_labels[i]].set(bo, use_median=True, median_size=4)

def transformPoints(points, origin=None, bump_offset=None, miter_angle=None, bevel_angle=None, 
                   slider_offset=None, crash_angle=None):
    """Transforms the Nx3 point array `points` according to the specified 
    parameters."""
    T = Transform()
    if origin is not None:
        T.translate(delx=-origin[0], dely=-origin[1], delz=-origin[2])
    if bump_offset is not None:
        T.translate(delx=-bump_offset[0], dely=-bump_offset[1], delz=-bump_offset[2])
    if slider_offset is not None:
        T.translate(dely=-slider_offset)
    if miter_angle is not None:
        T.rotate(psi=-miter_angle)
    if miter_angle is not None:
        T.rotate(phi=-bevel_angle)
    if crash_angle is not None:
        T.rotate(theta=-crash_angle)
    return T.transform(points)

def updateMiterAngle(markers: dict, ps: dict, n_frames: int=1):
    """Calculates the miter angle (radians) from the passed markers."""
    labels = ['miter_1', 'miter_2']
    for l in (labels):
        pts = markers[l].centers[-n_frames:]
        if len(pts) < 4: 
            return 0
        pts = transformPoints(miter_pts, ps['origin'].value, ps['bump_offset'].value)
        center = calcCOR(*pts)
        angle = calcAngleOfRotation(pts[-1], pts[-2], center)

    miter_pts = markers['miter_1'].centers[-n_frames:] + markers['miter_2'].centers[-n_frames:]
    miter_pts = transformPoints(miter_pts, ps['origin'].value, ps['bump_offset'].value)
    miter_cor = calcCOR(*miter_pts)
    miter_angle_1 = calcAngleOfRotation(miter_pts)

def calcCOR(positions: list) -> tuple:
    """Estimate the center of rotation given a series of positions of a rotated 
    point in 3D, where positions is a list of tuples representing the positions 
    (x, y, z)."""
    if len(positions) < 4:
        raise ValueError("At least four points required to estimate 3D center of rotation.")
    
    positions = np.array(positions)
    A, b = zip(*[getBisectorMatrices(*positions[i:i+2]) for i in range(len(positions) - 2)])
    A = np.array(A)
    b = np.array(b)
    center = np.linalg.lstsq(A, b, rcond=None)[0]
    return tuple(center)

def calcAngleOfRotation(pt1, pt2, cor) -> float:
    """Returns the angle of rotation between the two points centered on the 
    center of rotation (`cor`)."""
    vec1, vec2 = [np.array(pt) - cor for pt in [pt1, pt2]]
    vec1, vec2 = [np.linalg.norm(v) for v in [vec1, vec2]]
    dot = np.dot(vec1, vec2)
    theta = np.arccos(dot)
    return theta

def getBisectorMatrices(pt1, pt2, pt3):
    """Returns a linear algebraic structure for the bisector line extending from 
    the midpoint of the triangle connecting the three inputted points."""
    x1, y1, z1 = pt1
    x2, y2, z2 = pt2
    x3, y3, z3 = pt3
    
    # Midpoint of the triangle
    mid_x = np.mean([x1, x2, x3])
    mid_y = np.mean([y1, y2, y3])
    mid_z = np.mean([z1, z2, z3])
    
    # Normal vector to the plane defined by the triangle
    v1 = np.array([x2 - x1, y2 - y1, z2 - z1])
    v2 = np.array([x3 - x1, y3 - y1, z3 - z1])
    normal = np.cross(v1, v2) 
    norm = np.linalg.norm(normal)
    if norm == 0:
        raise ValueError("The given points are collinear in 3D space.")
    normal = normal / norm #unit normal vector for circle bisector
    related_pnt = np.dot(normal, [mid_x, mid_y, mid_z]) #related point on bisector
    return normal, related_pnt

def sendParamsToDB(params: dict, db: DBActor):
    """Stores the current 3D coordinates of the inputted parameters in the DB."""
    mocapseq_table = db.name.MOCAPSEQ_TBL_NAME
    seq_pk = db.getMaxPrimaryKey(mocapseq_table) + 1
    db.addEntry(mocapseq_table, [seq_pk], [db.name.MOCAPSEQ_TBL_COLS[0]])

    tbl_name = db.name.MOCAP_TBL_NAME
    col_names = db.name.MOCAP_TBL_COLS[1:]
    for p in params.values():
        values = [p.value, seq_pk, p.db_id]
        db.addEntry(tbl_name, values, col_names)

def setupMocapDB(param_labels: list, db: DBActor) -> dict:
    """Makes sure the DB contains all parameters and returns a dict of `Param` 
    objects with DB IDs, indexed by parameter label."""
    table_name = db.name.MOCAPMKR_TBL_NAME
    pk = db.getMaxPrimaryKey(table_name) + 1
    db_markers = db.getEntries(table_name)
    extant_labels = [e[1] for e in db_markers]
    params = dict()
    for label in param_labels:
        if label not in extant_labels:
            db.addEntry(table_name, [pk, label], db.name.MOCAPMKR_TBL_COLS)
            db_id = pk
            pk += 1
        else:
            db_id = db_markers[extant_labels.index(label.name)][0]
        params[label] = (Param(label, db_id))
    return params







################################
log.basicConfig(level=log.DEBUG)

class Mocap():
    """Performs linear transformations of planar points into 3D space."""
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

        # Setup marker tracking
        self.aruco = Aruco()
            
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
    
    def calibrate(self, calib_imgs: list, n_rows: int=4, n_cols: int=7, 
                  square_size: float=2.5):
        """Retrieves the calibration constants from a set of given images.
        
        Parameters
        ----------
        calib_imgs : list
            list of size 2, where each entry is a list of images taken from a 
            single camera. Each image is of a checkerboard captured by each 
            camera at a single instant (so that each image is synchronized). See
            `camera_interface.py` for assistance in taking the images.
        n_rows: int=4
            The number of rows of intersections on the checkerboard
        n_cols: int=7
            The number of columns of intersections on the checkerboard
        square_size: float=2.5
            The dimension of a single square on the checkerboard, in cm
        """
        calib = [calibCamIntrinsic(calib_imgs[i], n_rows, n_cols, square_size, 
                                   False) for i in range(len(calib_imgs))]
        mtx = [calib[i]['camera_matrix'] for i in range(len(calib))]
        dist = [calib[i]['dist_coeffs'] for i in range(len(calib))]
        rmse, R, T = stereoCalibration(*calib_imgs, *mtx, *dist, 4, 7, 2.5, False)
        projMat1 = mtx[0] @ cv.hconcat([np.eye(3), np.zeros((3,1))]) # Cam1 is the origin
        projMat2 = mtx[1] @ cv.hconcat([R, T]) # R, T from stereoCalibrate

        self.projection_mtrxs = [projMat1, projMat2]
        self.dist_coeffs = dist
        self.camera_mtrxs = mtx
    
    def getDefaultMarkers(self) -> list:
        """Returns list of planar coordinates of the default markers for the saw in 
        resting position."""
        labels = ['base_1', 'base_2', 'miter_1', 'bevel_1', 'arm_1', 'arm_2']
        coord = [[[761, 1231], [1061, 1354], [1343, 1360], [2162, 942], [1642, 618], [984, 15]],
                 [[1525, 668], [2102, 1028], [1054, 1503], [753, 1476], [627, 1341], [826, 29]]]
        
        markers = list()
        for i in range(len(labels)):
            markers.append(Marker(labels[i], [coord[0][i], coord[1][i]], (0,)*3))
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
            m.planar2global(self.camera_mtrxs, self.projection_mtrxs, self.dist_coeffs)
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
                self.getMarker(l).id = pk
            else:
                db_pk = db_markers[extant_labels.index(l)][0]
                self.getMarker(l).id = db_pk

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
 