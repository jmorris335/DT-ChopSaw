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
from src.auxiliary.geometry import pointDistance, shiftPoints
from src.auxiliary.transform import Transform
from src.db.actor import DBActor

log.basicConfig(level=log.DEBUG)

__slots__ = ("label", "db_id", "value")
class Param:
    """Container for a generic parameter to be inserted into the DB."""
    def __init__(self, label: str, db_id: int=-1, value: float=0.0):
        self.label = label
        self.db_id = db_id       
        self.value = value
    
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
    params = setupMocapDB(aruco.param_names, db)
    frame_counter = 0
    while all([cam.isOpened() for cam in cameras]):
        log.debug(f"Processing frame {frame_counter}")
        updateMarkers(cameras, proj_mtrxs, cam_mtrxs, dist_coefs, aruco)
        frame_counter += 1
        updateParamsFromMarkers(aruco, params)
        sendParamsToDB(params)

def updateMarkers(cameras: list, proj_mtrxs: list, cam_mtrxs: list, 
                  dist_coefs: list, aruco: Aruco) -> dict:
    """Updates 3D coordinates for each marker in the list found by both cameras."""
    imgs = [getImage(cam) for cam in cameras]
    if not all(imgs): #Check for invalid frames
        return 
    aruco.findMarkerCenters(imgs, proj_mtrxs, cam_mtrxs, dist_coefs)
    
def getImage(camera):
    """Returns the next frame from the given camera."""
    img = camera.read()
    if not img[0]:
        camera.release()
    return img[1]

def updateParamsFromMarkers(aruco: Aruco, params: dict):
    """Returns a dict of `Param` objects calculated from the aruco markers."""
    out_params = getChainValues(aruco.markers, params)
    labels = aruco.param_names
    for l, p in zip(labels, out_params):
        params[l].value = p

def getChainValues(markers: dict, params: dict):
    """Caller function that returns the values for the saw open chain.
    
    Parameters
    ----------
    markers : dict
        Dictionary of markers with centers specified in model coordinates.
        Must contain the following key, value pairs:
            miter : Marker
            bevel : Marker
            slider : Marker
            crash : Marker
    params : dict
        Dictionary of parameters that describe the saw. Must contain the 
        following key, value pairs:
            moffset_miter : tuple
            offset_stem : tuple
            moffset_bevel : tuple
            height_stem : float
            moffset_slider : tuple
            length_slider : float
            moffset_crash : tuple
        Note that moffset indicates the (x,y,z) distance from the marker center 
        to the ideal point of the attached joint.
    """
    miter = markers['miter']
    bevel = markers['bevel']
    slider = markers['slider']
    crash = markers['crash']
    moffset_miter = params['moffset_miter']
    offset_stem = params['ooffset_stem']
    moffset_bevel = params['moffset_bevel']
    height_stem = params['height_stem']
    moffset_slider = params['moffset_slider']
    length_slider = params['length_slider']
    moffset_crash = params['moffset_crash']

    miter_angle = findMiterAngle(miter, moffset_miter)
    bevel_COR = calcBevelCOR(miter_angle, offset_stem)
    bevel_angle = findBevelAngle(bevel, moffset_bevel, bevel_COR)
    stem_top = calcStemTop(miter_angle, offset_stem, bevel_angle, bevel_COR,
                           height_stem)
    slider_offset = findSliderOffset(slider, moffset_slider, stem_top)
    crash_COR = calcCrashCOR(miter_angle, offset_stem, bevel_angle, bevel_COR,
                             height_stem, slider_offset, length_slider)
    crash_zero = calcCrashZero(miter_angle, offset_stem, bevel_angle, bevel_COR,
                             height_stem, slider_offset, length_slider)
    crash_angle = calcCrashAngle(crash, moffset_crash, crash_zero, crash_COR)
    return miter_angle, bevel_angle, slider_offset, crash_angle

def calcAngle(pnt1: tuple, pnt2: tuple, intersection: tuple)-> float:
    """Returns the angle made by two vectors that intersect at `intersection` 
    and extend to the two inputted points."""
    v1, v2 = shiftPoints([pnt1, pnt2], intersection, inverse=True, copy=True)
    v1, v2 = [x / np.linalg.norm(x) for x in (v1, v2)]
    theta = np.arccos(np.dot(v1, v2))
    return theta

def adjustMarkerCenter(m: Marker, offset: tuple)-> tuple:
    """Returns the marker center, translated by the offset so as to be at the 
    correct location if the saw was at rest. This undoes any offset that might
    have occured when placing the marker."""
    center = shiftPoints(m.center[-1], offset, inverse=True)
    return center

def findMiterAngle(miter: Marker, offset: tuple)-> float:
    """Calculates the miter angle based on the miter marker."""
    marker_center = adjustMarkerCenter(miter, offset)
    miter_zero = (1, 0, 0)
    miter_COR = (0, 0, 0)
    return calcAngle(marker_center, miter_zero, miter_COR)

def calcBevelCOR(miter_angle, stem_offset, miter_COR=(0,0,0))-> tuple:
    """Calculates the model coordiantes for the bevel center of rotation for a 
    given miter angle."""
    T = Transform(centroid=miter_COR)
    T.translate(delx=-stem_offset)
    T.rotate(phi=miter_angle, x=miter_COR[0], y=miter_COR[1], z=miter_COR[2])
    bevel_COR = T.transform(miter_COR)[0][:3]
    return bevel_COR

def findBevelAngle(bevel: Marker, offset: tuple, bevel_COR: tuple)-> float:
    """Calculates the bevel angle based on the bevel marker."""
    bevel_center = adjustMarkerCenter(bevel, offset)
    bevel_zero = (0, 1, 0)
    bevel_angle = calcAngle(bevel_center, bevel_zero, bevel_COR)
    return bevel_angle

def calcStemTop(miter_angle, stem_offset, bevel_angle, bevel_COR, stem_height,
                miter_COR=(0,0,0))-> float:
    """Calculates the model coordinates for the top of the stem (bevel arm) for 
     a given miter and bevel angle."""
    T = Transform(centroid=miter_COR)
    T.translate(delx=-stem_offset)
    T.translate(dely=stem_height)
    T.rotate(psi=bevel_angle, x=bevel_COR[0], y=bevel_COR[1], z=bevel_COR[2])
    T.rotate(phi=miter_angle, x=miter_COR[0], y=miter_COR[1], z=miter_COR[2])
    stem_top = T.transform(miter_COR)[0][:3]
    return stem_top

def findSliderOffset(slider: Marker, offset: tuple, stem_top: tuple)-> float:
    """Calculates the distance from `stem_top` to the crash center of rotation.
    Note that the offset is the distance from the center of the slider center to 
    the crash arm center of rotation."""
    slider_center = adjustMarkerCenter(slider, offset)
    slider_offset = pointDistance(slider_center, stem_top)
    return slider_offset

def calcCrashCOR(miter_angle, stem_offset, bevel_angle, bevel_COR, stem_height, 
                 slider_offset, slider_length, miter_COR=(0,0,0))-> tuple:
    """Calculates the model coordinates for the crash center of rotation for a 
    given miter and bevel angle and slider offset."""
    T = Transform(centroid=miter_COR)
    T.translate(delx = -stem_offset)
    T.translate(dely = stem_height)
    T.translate(delx = slider_length + slider_offset)
    T.rotate(psi=bevel_angle, x=bevel_COR[0], y=bevel_COR[1], z=bevel_COR[2])
    T.rotate(phi=miter_angle, x=miter_COR[0], y=miter_COR[1], z=miter_COR[2])
    crash_COR = T.transform(miter_COR)[0][:3]
    return crash_COR

def calcCrashZero(miter_angle, stem_offset, bevel_angle, bevel_COR, stem_height, 
                 slider_offset, slider_length, miter_COR=(0,0,0))-> tuple:
    """Calculates the model coordinates for the zero postion of the crash arm for 
    a given miter and bevel angle and slider offset."""
    T = Transform(centroid=miter_COR)
    T.translate(delx = -stem_offset)
    T.translate(dely = stem_height)
    T.translate(delx = slider_length + slider_offset)
    T.translate(delx = 1)
    T.rotate(psi=bevel_angle, x=bevel_COR[0], y=bevel_COR[1], z=bevel_COR[2])
    T.rotate(phi=miter_angle, x=miter_COR[0], y=miter_COR[1], z=miter_COR[2])
    crash_zero = T.transform(miter_COR)[0][:3]
    return crash_zero

def calcCrashAngle(crash: Marker, offset: tuple, crash_zero: tuple, 
                   crash_COR: tuple)-> float:
    """Calculates the crash angle based on the crash arm marker."""
    crash_center = adjustMarkerCenter(crash, offset)
    crash_angle = calcAngle(crash_center, crash_zero, crash_COR)
    return crash_angle

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
        params[label] = Param(label, db_id)
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
 