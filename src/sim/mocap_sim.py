from src.mocap.mocap import startMocap

def mocapSim():
    # camera1 = "src/mocap/media/Saw_Synch_01.mp4"
    # camera2 = "src/mocap/media/Saw_Synch_02.mp4"
    camera1 = 1
    camera2 = 0

    calibration_imgs = [
        [f"src/mocap/media/calib/cam0_{i+1}.png" for i in range(10)],
        [f"src/mocap/media/calib/cam1_{i+1}.png" for i in range(10)]
    ]

    startMocap(camera1, camera2, calib_frames=calibration_imgs)

def scratchpaper():
    import cv2 as cv
    import numpy as np
    from src.mocap.calibration import calibCamIntrinsic, stereoCalibration
    calibration_imgs = [
        [f"src/mocap/media/calib/cam01_{i}.png" for i in range(12)],
        [f"src/mocap/media/calib/cam02_{i}.png" for i in range(12)]
    ]

    calib1 = calibCamIntrinsic(calibration_imgs[0], 4, 7, 2.5, False)
    calib2 = calibCamIntrinsic(calibration_imgs[1], 4, 7, 2.5, False)
    mtx1 = calib1['camera_matrix']
    mtx2 = calib2['camera_matrix']
    dist1 = calib1['dist_coeffs']
    dist2 = calib2['dist_coeffs']
    rmse, R, T = stereoCalibration(*calibration_imgs, mtx1, mtx2, dist1, dist2, 4, 7, 2.5, False)
    projMat1 = mtx1 @ cv.hconcat([np.eye(3), np.zeros((3,1))]) # Cam1 is the origin
    projMat2 = mtx2 @ cv.hconcat([R, T]) # R, T from stereoCalibrate

    # points1 is a (N, 1, 2) float32 from cornerSubPix
    points1 = np.array([[221,160]], 'float32')
    points2 = np.array([[100,381]], 'float32')

    points1u = cv.undistortPoints(points1, mtx1, dist1, None, mtx1)
    points2u = cv.undistortPoints(points2, mtx2, dist2, None, mtx2)

    points4d = cv.triangulatePoints(projMat1, projMat2, points1u, points2u)
    points3d = (points4d[:3, :]/points4d[3, :]).T

    print(tuple(points3d[0]))

