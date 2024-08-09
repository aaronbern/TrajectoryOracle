###############################################################################
#
# Author: Lorenzo D. Moon
# Professor: Dr. Anthony Rhodes
# Course: CS-441
# Assignment:  Final Project: Trajectory Oracle
# Description: Test the PlotYolo class and DetectedObject class
#
###############################################################################

from plot_yolo import DetectedObject, PlotYolo


def main():
    video_filepath = "./test_videos/one_man_walking.mp4"
    plot_yolo = PlotYolo(video_filepath)
    while True:
        frame = plot_yolo.get_next_frame()
        if frame is None:
            break
        plot_yolo.save_frame(frame)
    return


if __name__ == "__main__":
    main()
