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

import sys
import threading
import time

import cv2
import numpy as np
from ultralytics import YOLO


class PlotYolo:
    def __init__(self, video_filepath, distance_threshold=50):
        self.yolo = YOLO("yolov8s.pt")
        self.video_filepath = video_filepath
        self.video_capture = cv2.VideoCapture(self.video_filepath)
        self.objects_last_frame = []
        self.objects = []
        self.frames_processed = 0
        self.__frame_size = None
        self.distance_threshold = distance_threshold
        self.used_colors = {}
        self.set_frame_size()
        pass

    def __del__(self):
        # Zero out all data, and release the video capture,
        # and close all windows
        self.video_filepath = None
        self.objects_last_frame = None
        self.objects = None
        self.frames_processed = 0
        self.__frame_size = None
        self.distance_threshold = None
        self.video_capture.release()
        # Nothing should open a window, but if it ends up being altered to
        # create a window this is a safety measure.
        cv2.destroyAllWindows()
        pass

    @property
    def object_count(self):
        # Return the number of objects detected in the current frame
        return len(self.objects)

    @property
    def frame_size(self):
        # Return the size of the frame
        return self.__frame_size

    def set_frame_size(self):
        # Get the frame size of the video
        if self.__frame_size is None:
            ret, frame = self.video_capture.read()
            self.__frame_size = frame.shape[:2]
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        return

    def get_next_frame_raw(self):
        # Return the next frame and the results of the YOLO detection
        # This is a raw version of the function, and should not be used
        # outside of the class.
        try:
            frame, results = next(self.yield_next_frame_raw())
        except StopIteration:
            return None

        return frame, results

    def gen_unique_color(self):
        # Rengerate a random number 100-250 must be int
        red = np.random.randint(20, 255)
        green = np.random.randint(20, 255)
        blue = np.random.randint(20, 255)

        # If all of them aren't 255, make one randomly 255
        if red != 255 and green != 255 and blue != 255:
            rand = np.random.randint(0, 3)
            if rand == 0:
                red = 255
            elif rand == 1:
                green = 255
            else:
                blue = 255

        if (red, green, blue) in self.used_colors:
            return self.gen_unique_color()

        self.used_colors[(red, green, blue)] = True
        return tuple([red, green, blue])

    def yield_next_frame_raw(self):
        # Yield the next frame and the results of the YOLO detection
        # This is a raw version of the function, and should not be used
        # outside of the class.
        while True:
            ret, frame = self.video_capture.read()
            self.frames_processed += 1
            if not ret:
                return None, None
            results = self.yolo(frame, verbose=False)
            yield frame, results

    def save_frame(self, frame):
        # Save the frame to the test_output folder
        # This was used for debugging purposes
        cv2.imwrite(f"./test_output/frame_{self.frames_processed}.jpg", frame)
        return

    def get_next_frame(self, confidence=0.6):
        # Get the next frame and the detected objects
        # This function will return the detected objects in the frame
        # that have a confidence greater than the threshold.
        # The threshold is set to 0.6 by default.
        try:
            # Get the raw data, and unpack it
            frame, results = self.get_next_frame_raw()
        except TypeError:
            return None

        objects = []
        object_id = 0

        for result in results:
            classes_names = result.names

            # Loop through the detected objects
            # If the confidence is greater than the threshold, add it to the list
            # of objects found this frame
            for box in result.boxes:
                if box.conf[0] > confidence:
                    [x1, y1, x2, y2] = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    cls = int(box.cls[0])
                    class_name = classes_names[cls]
                    # color = self.getColors(cls)

                    id = f"{self.frames_processed}_{object_id}"

                    objects.append(
                        DetectedObject(
                            id,
                            class_name,
                            cls,
                            box.conf[0],
                            x1,
                            y1,
                            x2,
                            y2,
                        )
                    )

                    object_id += 1

        
        # Move last frames objects to a holding list
        self.objects_last_frame = self.objects

        # Set the current objects to the new objects
        self.objects = objects

        # Correlate the objects between the last frame and the current frame
        # This will assign an ID to the object if it is the same object
        # from the last frame.
        self.correlate_objects()

        # Return all the objects detected this frame, and their ID's
        return self.objects

    def correlate_objects(self):
        # Correlate the objects between the last frame and the current frame
        # This will assign an ID to the object if it is the same object
        # from the last frame.
        for obj in self.objects:
            closest_match = 100000000
            for obj_last in self.objects_last_frame:
                # If the classes are different, move on
                if obj_last.class_id != obj.class_id:
                    continue

                # L2 distance between the centers of the two objects.
                # This might get expensive if there are a lot of objects.
                distance = (
                    (obj.x - obj_last.x) ** 2 + (obj.y - obj_last.y) ** 2
                ) ** 0.5

                if distance < closest_match:
                    closest_match = distance
                    if distance < self.distance_threshold:
                        obj.id = obj_last.id
                        obj.last_position = obj_last.position
                        break
        return

    def spinning_bar(self, stop_event):
        # This is used to show a spinning bar while the video is being processed.
        # This was developed on a computer with a slower processor, and might
        # not be needed on a better machine, that's why plot_video has a
        # spinner=False parameter.
        spinner = ["|", "/", "-", "\\"]
        idx = 0

        while not stop_event.is_set():
            sys.stdout.write(f"\r{spinner[idx]}")
            sys.stdout.flush()
            time.sleep(0.1)  # Adjust the speed of the spinner
            idx = (idx + 1) % len(spinner)

        sys.stdout.write("\rDone!\n")

    def plot_video(self, spinner=False):
        # Run the spinning bar in a separate thread
        if spinner:
            stop_event = threading.Event()
            spinner_thread = threading.Thread(
                target=self.spinning_bar, args=(stop_event,)
            )
            spinner_thread.start()
        colors = {}

        # Create a white image to draw the trajectory on
        frame = np.ones((self.frame_size[0], self.frame_size[1], 3), np.uint8) * 255

        while True:
            objects = self.get_next_frame()
            if objects is None:
                break

            for obj in objects:
                if obj.id not in colors:
                    colors[obj.id] = self.gen_unique_color()

                # Draw a circle at the current position
                cv2.circle(
                    frame,
                    obj.position,
                    5,
                    colors[obj.id],
                    -1,
                )

                if obj.last_position is not None:
                    continue

        # Save the frame
        video_filename = self.video_filepath.split("/")[-1]
        video_filename = video_filename.split(".")[0]
        cv2.imwrite(
            f"./test_output/{video_filename}_dt_{self.distance_threshold}.jpg", frame
        )

        if spinner:
            stop_event.set()
            spinner_thread.join()
        return


class DetectedObject:
    # This class is used to store the detected objects
    # This will what the Reinforcement Learning agent will use
    # to track the object.
    def __init__(self, id, class_name, class_id, confidence, x1, y1, x2, y2):
        self.id = id
        self.class_name = class_name
        self.class_id = class_id
        self.confidence = confidence
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.x = (x1 + x2) // 2
        self.y = (y1 + y2) // 2
        self.w = x2 - x1
        self.h = y2 - y1
        self.last_position = None
        pass

    @property
    def position(self):
        return (self.x, self.y)

    def __str__(self):
        string = f"ID: {self.id}: {self.class_name}@({self.x}, {self.y})"
        return string
