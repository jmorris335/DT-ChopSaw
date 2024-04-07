"""
| File: calibration.py 
| Info: Functions for calibrating a stereo camera setup for motion capture using Direct Linear
|   Transformation.
| Author: John Morris, jhmrrs@clemson.edu  
| Organization: Product Lifecycle Management Center at Clemson University, plmcenter@clemson.edu  
| Permission: Copyright (C) 2023, John Morris. All rights reserved. Should not be reproduced, 
|   edited, sourced, or utilized without written permission from the author or organization
| Version History:
| - 0.0, 13 Mar 2024: initialized
"""

import numpy as np
import cv2 as cv


def calibrateCameraIntrinsic(image_filepaths: list, num_rows: int, num_cols: int, square_size: float, show_images: bool=False) -> dict:
    """Calibrates the camera against several checkerboard images using the OpenCV library.

    This calibration focuses on intrisic calibration (not sychronization). The values 
    discovered can be reused for the same camera in any position.

    Parameters
    ----------
    image_filepaths: list
        List of strings with each entry denoting a filepath for an image to use in 
        calibration.
    show_images: bool, default=False
        Function will show each calibration image and provide skipping options if set to
        true.

    Returns
    -------
    dict: dictionary of values with the following keywords (with descriptions taken from the 
        OpenCV documentation: https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#ga91018d80e2a93ade37539f01e6f07de5):
        - 'ret_val': The overall RMS re-projection error.
        - 'camera_matrix': Input/output 3x3 floating-point camera intrinsic matrix. 
           If CALIB_USE_INTRINSIC_GUESS and/or CALIB_FIX_ASPECT_RATIO, CALIB_FIX_PRINCIPAL_POINT 
           or CALIB_FIX_FOCAL_LENGTH are specified, some or all of fx, fy, cx, cy must be 
           initialized before calling the function.
        - 'dist_coeffs': Input/output vector of distortion coefficients.

    Sources
    -------
    Function modified from original script by Temuge Batpurev (2023), available at 
    https://github.com/TemugeB/python_stereo_camera_calibrate
    """
    #read all frames
    images = [cv.imread(img_path, 1) for img_path in image_filepaths]

    #Coordinates of squares in the checkerboard world space
    object_pnt = calcCheckerboardCoordinates(num_rows, num_cols, square_size)

    height, width, depth = images[0].shape  #Frame dimensions. Frames should be the same size.

    img_pnts = list() # 2d points in image plane.
    obj_pnts = list() # 3d point in real world space

    for frame in images:
        corners = findCheckerboardCorners(frame, num_rows, num_cols, show_images)
        if corners is not None:
            obj_pnts.append(object_pnt)
            img_pnts.append(corners)

    cv.destroyAllWindows()
    ret, cmtx, dist, rvecs, tvecs = cv.calibrateCamera(obj_pnts, img_pnts, (width, height), None, None)
    return {'rmse': ret, 'camera_matrix': cmtx, 'dist_coeffs': dist}

def calcCheckerboardCoordinates(num_rows: int, num_cols: int, square_size: float):
    """Calcuates the (x, y, z) coordinates for a checkerboard calibration pattern.
    """
    coordinates = np.zeros((num_rows*num_cols,3), np.float32)
    coordinates[:,:2] = np.mgrid[0:num_rows,0:num_cols].T.reshape(-1,2)
    coordinates = square_size * coordinates
    return coordinates

def findCheckerboardCorners(frame, num_rows, num_cols, show_images: bool=False):
    """Returns the corners for a checkerboard in the frame.

    Utilizes the OpenCV library. See calibrateCameraIntrinsic for more descriptions
    """
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    pattern_found, corners = cv.findChessboardCorners(gray, (num_rows, num_cols), None)

    if pattern_found == True:
        corners = improveCornerDetection(gray, corners)
        if show_images:
            if displayCalibImg(frame, num_rows, num_cols, corners) is None:
                return None
            
    return corners

def improveCornerDetection(frame, corners, win_size : tuple=(11, 11)):
    """Improves corner detection to sub-pixel accuracy.

    Parameters
    ----------
    frame : array_like
        The image with detected checkerboard pattern.
    corners : list
        List of pixal (2D) coordinates for initial corners detected.
    win_size : tuple, default=(11, 11)
        Half of the side length of the search window for the best corner, in pixels. 
        Smaller is higher accuracy but can mess corners and run more slowly.
    
    Calls cornerSubPix from OpenCV: 
    https://docs.opencv.org/4.x/dd/d1a/group__imgproc__feature.html#ga354e0d7c86d0d9da75de9b9701a9a87e
    """
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv.cornerSubPix(frame, corners, win_size, (-1, -1), criteria)
    return corners

def displayCalibImg(frame, num_rows, num_cols, corners):
    """Displalys the calibration images and detected checkerboard corners.

    Utilizes the OpenCV library.
    
    Returns
    -------
    None if the image is skipped, else the inputted frame.
    """
    cv.drawChessboardCorners(frame, (num_rows,num_cols), corners, True)
    cv.putText(frame, 'Press "s" to skip this sample', (25, 25), cv.FONT_HERSHEY_COMPLEX, 1, (0,255,255), 1)
    cv.imshow('img', frame)
    k = cv.waitKey(0)

    if k & 0xFF == ord('s'):
        print('skipping')
        return None
    
    return frame

def stereoCalibration(img_filepaths_A: list, img_filepaths_B: list,
                      mtx_A, mtx_B, dist_A, dist_B,
                      num_rows: int, num_cols: int, square_size: float, show_images: bool=False):
    #open images
    imgs_A = [cv.imread(img_path, 1) for img_path in img_filepaths_A]
    imgs_B = [cv.imread(img_path, 1) for img_path in img_filepaths_B]

    #coordinates of squares in the checkerboard world space
    object_pnt = calcCheckerboardCoordinates(num_rows, num_cols, square_size)

    #frame dimensions. Frames should be the same size.
    height, width, depth = imgs_A[0].shape

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    stereocalibration_flags = cv.CALIB_FIX_INTRINSIC

    #Coordinates of checkerboards
    img_pts_A, img_pts_B  = list(), list() #Pixel coordinates
    object_pts = list() # 3d point in real world space

    for img_A, img_B in zip(imgs_A, imgs_B):
        corners_A = findCheckerboardCorners(img_A, num_rows, num_cols, show_images)
        corners_B = findCheckerboardCorners(img_B, num_rows, num_cols, show_images)
        
        if (corners_A is not None) and (corners_B is not None):
            object_pts.append(object_pnt)
            img_pts_A.append(corners_A)
            img_pts_B.append(corners_B)

    rmse, CM1, dist0, CM2, dist1, R, T, E, F = cv.stereoCalibrate(object_pts, img_pts_A, img_pts_B, mtx_A, dist_A,
                                                                 mtx_B, dist_B, (width, height), criteria = criteria, 
                                                                 flags = stereocalibration_flags)
    
    cv.destroyAllWindows()
    return rmse, R, T

if __name__ == '__main__':
    # cameras = ["00", "01"]
    cameras = ["A", "B"]
    num_rows = 4
    num_cols = 7
    square_size = 2.5 #Units are cm
    cam_mtrxs, cam_dist_coeffs = list(), list()
    for camera in cameras:
        # mono_img_paths = [f"src/mocap/test-mocap/{camera}_calib_{i+1}.jpg" for i in range(9)]
        mono_img_paths = [f"src/mocap/temugeB_demo/frames/mono_calib/camera{camera}_{i}.png" for i in range(4)]
        calib_consts = calibrateCameraIntrinsic(mono_img_paths, num_rows, num_cols, square_size, show_images=False)
        print(f"RMSE for intrinsic calibration of camera {camera} is {calib_consts['rmse']}")
        cam_mtrxs.append(calib_consts['camera_matrix'])
        cam_dist_coeffs.append(calib_consts['dist_coeffs'])
    
    stereo_img_paths_A = [f"src/mocap/temugeB_demo/frames/synched/cameraA_{i}.png" for i in range(4)]
    stereo_img_paths_B = [f"src/mocap/temugeB_demo/frames/synched/cameraB_{i}.png" for i in range(4)]
    rmse, R, T = stereoCalibration(stereo_img_paths_A, stereo_img_paths_B, *cam_mtrxs, *cam_dist_coeffs, 
                      num_rows, num_cols, square_size, show_images=False)
    print(f"RMSE for stereo calibration is {rmse}")

