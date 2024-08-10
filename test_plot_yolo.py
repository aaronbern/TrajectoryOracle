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
    one_walking = "./test_videos/one_man_walking.mp4"
    many_walking = "./test_videos/many_walking.mp4"
    video_filepath = many_walking
    #video_filepath = one_walking
    plot_yolo = PlotYolo(video_filepath)
    plot_yolo.plot_video(spinner=True)
    # print(f"Frame size: {plot_yolo.frame_size}")
    # while True:
    #   frame = plot_yolo.get_next_frame()
    #   if frame is None:
    #       exit(0)
    #
    #   for obj in plot_yolo.objects:
    #       print(f"{obj}")

    return


if __name__ == "__main__":
    main()
