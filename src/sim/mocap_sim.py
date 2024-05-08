from src.mocap.mocap import Mocap

import cv2 as cv

def saw_sim():
    # image1A = "src/mocap/mocap_images/00_pos_1_highlighted.png"
    # image1B = "src/mocap/mocap_images/01_pos_1_highlighted.png"

    camera1 = "src/mocap/media/Saw_Synch_01.mp4"
    camera2 = "src/mocap/media/Saw_Synch_02.mp4"
    calibration_imgs = [
        [f"src/mocap/media/calib/cam01_{i}.png" for i in range(12)],
        [f"src/mocap/media/calib/cam02_{i}.png" for i in range(12)]
    ]

    mocap = Mocap(camera1, camera2, calibration_imgs)
    print(mocap.planar2global(*mocap.planar_markers))