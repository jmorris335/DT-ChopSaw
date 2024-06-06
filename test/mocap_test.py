import unittest

from src.mocap.calibration import *

class MocapTest(unittest.TestCase):
    @classmethod
    def setUpClass(self) -> None:
        camera = "00"
        self.img_paths = [f"src/mocap/test-mocap/{camera}_calib_{i+1}.jpg" for i in range(9)]
        self.num_rows = 4
        self.num_cols = 7
        self.square_size = 2.54

    def test_calibration(self):
        rmse, cam_mtx, dist = calibCamIntrinsic(self.img_paths, self.num_rows, self.num_cols, self.square_size)
        assert(rmse < 0.5)

if __name__ == '__main__':
    t = MocapTest()
    t.test_calibration()