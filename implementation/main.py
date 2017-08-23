import sys

import imageio
import matplotlib.pyplot as plt

from classifier import Classifier
from configuration import Configuration
from helper import Helper
from lane_detection import LaneDetection
from pre_processing import PreProcessing
from window_search import WindowSearch

config = Configuration().__dict__
sys.path.append("implementation/")


def __main__():
    # get video stream
    video_cap = imageio.get_reader(config["project_video"])
    # polynomial lane fit
    lanes_fit = []

    # classifier and scaler
    classifier = Classifier.get_trained_classifier(use_pre_trained=False)

    # load calibration parameters:
    camera_matrix, dist_coef = PreProcessing.load_calibration_params()
    for index, img in enumerate(video_cap):
        if index % 2 == 0:
            # get lanes
            lanes_fit, img = LaneDetection.pipeline(img, lanes_fit, camera_matrix, dist_coef)
            # resize image to improve speed of vehicle detection using classifier

            # jpg to png
            if config["is_training_png"]:
                img = Helper.scale_to_png(img)

            # 3 channel without alpha
            img = img[:, :, :3]
            bounding_boxes = []
            # get bounding boxes for left side
            x_start_stop_left, y_start_stop_left = config["xy_start_stop_left"]
            bounding_boxes += WindowSearch.get_bounding_boxes(img, classifier,
                                                              x_start_stop_left,
                                                              y_start_stop_left)
            # # get bounding boxes for top side
            x_start_stop_top, y_start_stop_top = config["xy_start_stop_top"]
            bounding_boxes += WindowSearch.get_bounding_boxes(img, classifier,
                                                              x_start_stop_top,
                                                              y_start_stop_top)
            # get bounding boxes for right side
            x_start_stop_right, y_start_stop_right = config["xy_start_stop_right"]
            bounding_boxes += WindowSearch.get_bounding_boxes(img, classifier,
                                                              x_start_stop_right,
                                                              y_start_stop_right)
            
            # remove false positives and duplicates from detection
            detected_cars = Helper.remove_false_positives(img, bounding_boxes)

            # visualization
            plt.imshow(detected_cars)
            plt.pause(0.0001)


__main__()
