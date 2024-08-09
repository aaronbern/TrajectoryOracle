###############################################################################
#
# Author: Lorenzo D. Moon
# Professor: Dr. Anthony Rhodes
# Course: CS-441
# Assignment:  Final Project: Trajectory Oracle
# Description: This class will take in a video file, and output a plot of the
#              trajectory of the object(s) in the video. The object will be
#              detected using YOLOv8, and output in a format that can be used
#              by the "TrajectoryOracle" agent.
#
###############################################################################

import cv2
from ultralytics import YOLO


class PlotYolo:
    def __init__(self, video_filepath, visual_output=False):
        self.yolo = YOLO("yolov8s.pt")
        self.video_filepath = video_filepath
        self.video_capture = cv2.VideoCapture(self.video_filepath)
        self.objects_last_frame = []
        self.objects = []
        self.frames_processed = 0
        pass

    def __del__(self):
        self.video_capture.release()
        cv2.destroyAllWindows()
        pass

    def get_next_frame(self):
        self.objects_last_frame = self.objects
        self.objects = []

        try:
            frame, results = next(self.yield_next_frame())
        except StopIteration:
            return None

        for result in results:
            # get the classes names
            classes_names = result.names

            # iterate over each box
            for box in result.boxes:
                # check if confidence is greater than 40 percent
                if box.conf[0] > 0.4:
                    # get coordinates
                    [x1, y1, x2, y2] = box.xyxy[0]
                    # convert to int
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    # get the class
                    cls = int(box.cls[0])

                    # get the class name
                    class_name = classes_names[cls]

                    # get the respective colour
                    colour = self.getColors(cls)

                    # draw the rectangle
                    cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

                    # put the class name and confidence on the image
                    cv2.putText(
                        frame,
                        f"{classes_names[int(box.cls[0])]} {box.conf[0]:.2f}",
                        (x1, y1),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        colour,
                        2,
                    )
        return frame

    def getColors(self, cls_num):
        base_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        color_index = cls_num % len(base_colors)
        increments = [(1, -2, 1), (-2, 1, -1), (1, -1, 2)]
        color = [
            base_colors[color_index][i]
            + increments[color_index][i] * (cls_num // len(base_colors)) % 256
            for i in range(3)
        ]
        return tuple(color)

    def yield_next_frame(self):
        while True:
            ret, frame = self.video_capture.read()
            self.frames_processed += 1
            if not ret:
                return None
            results = self.yolo(frame)
            yield frame, results

    def save_frame(self, frame):
        cv2.imwrite(f"./test_output/frame_{self.frames_processed}.jpg", frame)
        return


class DetectedObject:
    def __init__(self, obj_id, class_name, class_id, confidence, x, y, w, h):
        self.obj_id = obj_id
        self.class_name = class_name
        self.class_id = class_id
        self.confidence = confidence
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        pass

    def __str__(self):
        string = f"Object {self.obj_id} of class {self.class_name} at"
        string += f" ({self.x}, {self.y}) with confidence {self.confidence}"
        return string
