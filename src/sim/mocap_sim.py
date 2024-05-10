from src.mocap.mocap import Mocap

def mocapSim():
    camera1 = "src/mocap/media/Saw_Synch_01.mp4"
    camera2 = "src/mocap/media/Saw_Synch_02.mp4"

    calibration_imgs = [
        [f"src/mocap/media/calib/cam01_{i}.png" for i in range(12)],
        [f"src/mocap/media/calib/cam02_{i}.png" for i in range(12)]
    ]

    mocap = Mocap(camera1, camera2, calib_frames=calibration_imgs)
    # mocap.start()
    scratchpaper(mocap)

def scratchpaper(mocap: Mocap):
    import cv2 as cv
    import numpy as np
    prj_mtrxs = mocap.projection_mtrxs
    cv.projectPoints(np.array([[0,0,39]], np.float), 
                        right_rvecs, 
                        right_tvecs,
                        prj_mtrxs[0],
                        right_distortion)

