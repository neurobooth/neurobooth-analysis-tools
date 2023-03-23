"""
Preprocessing functions related to Intel RealSense .bag files.
Ideally, anything requiring pyrealsense2 should be isolated to this file so that the dependency can be ignored when not
working with .bag files.
"""


import numpy as np
import cv2
import pyrealsense2 as rs
from neurobooth_analysis_tools.data.files import resolve_filename, FILE_PATH


def bag2avi(bag_file: FILE_PATH, avi_file: FILE_PATH) -> None:
    """
    Extract the color information from the .bag file and save it as a .avi file.
    :param bag_file: The input .bag file to extract color frames from
    :param avi_file: The output .avi file to write to
    """
    bag_file = resolve_filename(bag_file)
    avi_file = resolve_filename(avi_file)

    # Set up the .bag file for streaming
    config = rs.config()
    rs.config.enable_device_from_file(config, bag_file, repeat_playback=False)
    config.enable_stream(rs.stream.color)
    pipeline = rs.pipeline()
    pipeline_profile = pipeline.start(config)
    pipeline_profile.get_device().as_playback().set_real_time(False)

    # Retrieve characteristics of the .bag file needed to create the .avi file
    stream = pipeline_profile.get_stream(rs.stream.color)
    fps = stream.fps()
    intrinsics = stream.as_video_stream_profile().get_intrinsics()
    width, height = intrinsics.width, intrinsics.height

    # Write the .avi file
    video_out = cv2.VideoWriter(avi_file, cv2.VideoWriter_fourcc(*'MJPG'), fps, (width, height))
    while True:
        success, frame = pipeline.try_wait_for_frames(timeout_ms=1)
        if not success:
            break
        frame = np.asanyarray(frame.get_color_frame().get_data())
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_out.write(frame)
    video_out.release()
    pipeline.stop()
