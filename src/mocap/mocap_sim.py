from src.mocap.mocap import Mocap

def saw_sim():
    image1A = "src/mocap/mocap_images/00_pos_2.jpg"
    image1B = "src/mocap/mocap_images/01_pos_2.jpg"
    image2A = "src/mocap/mocap_images/00_pos_3.jpg"
    image2B = "src/mocap/mocap_images/01_pos_3.jpg"

    mocap = Mocap(num_cameras=2)
    mocap.updateSkeleton([image1A, image1B])
    mocap.updateSkeleton([image2A, image2B])