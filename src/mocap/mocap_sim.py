from src.mocap.mocap import Mocap

import cv2 as cv

def saw_sim():
    image1A = "src/mocap/mocap_images/00_pos_1_highlighted.png"
    image1B = "src/mocap/mocap_images/01_pos_1_highlighted.png"

    mocap = Mocap(num_cameras=2)
    mocap.updateSkeleton([image1A, image1B])
    print(mocap.planar2global())