import numpy as np
import av
import io

class Singleton:
    __instance = None

    @staticmethod
    def getInstance(): 
        """
        Static access method
        :return: Singleton instance
        """
        if Singleton.__instance == None:
            Singleton()
        return Singleton.__instance

    def __init__(self, height, width):
        """ 
        Virtually private constructor 
        :param height: height of the video
        :param width: width of the video
        """
        self.output_memory_file = io.BytesIO()
        self.output = av.open(self.output_memory_file, mode='w', format='mp4') # Open "in memory file as MP4"
        self.stream = self.output.add_stream('h264', 30) # Add a video stream with 30 fps
        self.stream.height = height
        self.stream.width = width
        self.stream.pix_fmt = 'yuv420p'
        self.stream.options = {'crf': '0', 'preset': 'ultrafast'}

    def write(self, frame):
        """
        Write a frame to the mp4 file
        :param frame: frame to write
        """
        frame = av.VideoFrame.from_ndarray(frame, format='bgr24') # convert image from NumPy array to frame
        packet = self.stream.encode(frame) # encode video frame
        self.output.mux(packet) # add encoded frame to MP4 file

    def close_and_get(self, output_path):
        """
        Close the mp4 file and write to output path
        :param output_path: path to write the mp4 file
        """
        packet = self.stream.encode(None) # flush encoder
        self.output.mux(packet) # indicate end of stream
        self.output.close() # close MP4 file

        # write BytesIO from RAM to file
        with open(output_path, "wb") as f:
            f.write(self.output_memory_file.getbuffer())


def get_frames(video_path):
    """
    Get frames from a video
    :param video_path: path to video
    :return: generator of frames
    """
    container = av.open(video_path) # open video
    stream = container.streams.video[0] # get video stream
    for frame in container.decode(stream): # decode video stream
        yield frame.to_ndarray(format='bgr24') # convert frame to NumPy array

    
    