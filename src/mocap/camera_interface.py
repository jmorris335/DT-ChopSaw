"""
| File: camera_interface.py 
| Info: Presents class that abstracts interfacing with a camera using OpenCV
| Author: John Morris, jhmrrs@clemson.edu  
| Organization: Product Lifecycle Management Center at Clemson University, plmcenter@clemson.edu  
| Permission: Copyright (C) 2023, John Morris. All rights reserved. Should not be reproduced, edited, sourced, or utilized without written permission from the author or organization

| Version History:
| - 0.0, 26 Apr 2024: Initialized
"""

import cv2 as cv
from threading import Thread

class VideoInterface:
    """
    Abstracts interfacing with two cameras for stereographic imaging using the 
    OpenCV library.

    Parameters
    ----------
    camera_ids : list
        List of ids for each camera, where each entry is (int | str | 
        cv.VideoCap[ture]). If the camera is a USB camera, then the id is passed 
        as an int (usually 0 or 1). If the camera id is a video file, then the 
        path to the file should be passed as a string. The id can also be a 
        previously setup VideoCapture class.
    dir_path : str, default='src/mocap/media'
        The directory where video frames should be written to.
    """
    def __init__(self, camera_ids: list, dir_path: str= "src/mocap/media"):
        self.num_cams = len(camera_ids)
        self.dir_path = dir_path

        self.vcs = list()
        for id in camera_ids:
            if isinstance(id, cv.VideoCapture):
                self.vcs.append(id)
            else:
                self.vcs.append(cv.VideoCapture(id))

    def __del__(self):
        for vc in self.vcs:
            vc.release()
        cv.destroyAllWindows()

    def getCalibrationFrames(self, num_frames: int=10)-> list:
        """Records 10 frames (after user input) for use in camera calibration.
        
        Each frame should include a chessboard grid. Frames are written to the 
        class directory. Function returns a list of filepaths to written images.
        """
        frames_taken = 0
        file_paths = [list(), list()]
        while self.camerasOpen() and (frames_taken < num_frames):
            framesRead, frames = self.readFrames()

            while all(framesRead) and (frames_taken < num_frames):
                self.flipFrames(frames)
                msg = f'Sequence {frames_taken + 1} / {num_frames}; Press spacebar to capture, press any other key to retake.' 
                self.placeMessage(frames[0], msg)
                self.showFrames(frames)

                k = cv.waitKey(0)
                if k % 256 == 32: # SPACEBAR pressed
                    for i,frame in enumerate(frames):
                        path = self.dir_path + f'/calib/cam{i}_{frames_taken+1}.png'
                        cv.imwrite(path, frame)
                        file_paths[i].append(path)
                    frames_taken += 1
                else:
                    break

        return file_paths

    def writeVideoStream(self, fourcc: str='mp4v', filetype: str='mp4', 
                         framerate: float=20.0):
        """Records a video simulateously from all cameras. 
        
        Parameters
        ----------
        fourcc : str, default='mp4v'
            The 4 character code for the video codec (encoder/decoder) to be used.
        filetype : str, default='mp4'
            String for the filetype for a encoded video, as in `movie.<filetype>`.
        framerate : float, default=20.0
            Frame rate (frames/second) of the encoded video.
        """
        FOURCC = cv.VideoWriter_fourcc(*fourcc)
        vws = self.makeVideoWriters(filetype, FOURCC, framerate, self.num_cams)
        keyval = 0

        while(self.camerasOpen()): #Press spacebar to quit
            framesRead, frames = self.readFrames()

            if all(framesRead):
                self.flipFrames(frames)
                self.showFrames(frames)
                for vw, frame in zip(vws, frames):
                    vw.write(frame)

            keyval = cv.waitKey(1)
            if (keyval % 256) == 32:
                for vw in vws:
                    vw.release()
                return

    def makeVideoWriters(self, filetype, FOURCC, framerate, num_cams):
        """Returns a list of video writers for each camera."""
        frame_dims = [[int(vc.get(i)) for i in [3, 4]] for vc in self.vcs]
        names = [self.dir_path + '/saw_synch_' + str(i) + '.' + filetype for i in range(num_cams)]
        vws = [cv.VideoWriter(names[i], FOURCC, framerate, (frame_dims[i][0],
                                   frame_dims[i][1])) for i in range(num_cams)]
        return vws

    def placeMessage(self, frame, message: str, origin: tuple=(50, 50), 
                     fontFace=cv.FONT_HERSHEY_DUPLEX, fontScale: float=0.75, 
                     color=(255,150,100), thickness: int=2, lineType=cv.LINE_AA):
        """Places the text in `message` on the plot. Wrapper for `cv.putText()`"""
        cv.putText(frame, message, origin, fontFace, fontScale, color, thickness, lineType) 
    
    def readFrames(self):
        """Reads the frames from all cameras in the class."""
        readFrames, frames = [a for a in zip(*[vc.read() for vc in self.vcs])]
        return readFrames, frames

    def camerasOpen(self):
        """Returns true if all cameras are open."""
        return all([vc.isOpened() for vc in self.vcs])
    
    def flipFrames(self, frames, direction: int=1):
        """Flips all frames along the specified direction."""
        frames = [cv.flip(frame, direction) for frame in frames]
        return frames
    
    def showFrames(self, frames):
        """Shows each frame. Wrapper for `cv.imshow`"""
        for i,frame in enumerate(frames):
            cv.imshow(f'Camera {i}', frame)

if __name__ == '__main__':
    VI = VideoInterface([0, 1])
    # VI.getCalibrationFrames()
    VI.writeVideoStream()