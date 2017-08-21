import sys

import cv2 as cv
import imageio

from classifier import Classifier
from configuration import Configuration
from helper import Helper
from lane_detection import LaneDetection
from pre_processing import PreProcessing
from visualization import Visualization
from window_search import WindowSearch

config = Configuration().__dict__
sys.path.append("implementation/")


def __main__():
    # get video stream
    video_cap = imageio.get_reader(config["project_video"])
    # polynomial lane fit
    lanes_fit = []

    # classifier and scaler
    classifier = Classifier.get_trained_classifier(use_pre_trained=True)

    # load calibration parameters:
    camera_matrix, dist_coef = PreProcessing.load_calibration_params()
    for img in video_cap:
        # get lanes
        lanes_fit, img = LaneDetection.pipeline(img, lanes_fit, camera_matrix, dist_coef)

        # resize image to improve speed of vehicle detection using classifier
        img = cv.resize(img, None, fx=0.8, fy=0.8, interpolation=cv.INTER_LANCZOS4)

        # jpg to png
        if config["is_training_png"]:
            img = Helper.scale_to_png(img)

        # 3 channel without alpha
        img = img[:, :, :3]

        # image dimensions
        width, height = img.shape[1], img.shape[0]

        # region of interest (ROI)
        # lower half of the image
        y_start_stop = [int(height / 2), height]

        # get bounding boxes for cars in the image
        bounding_boxes = WindowSearch.get_bounding_boxes(img, classifier, y_start_stop)

        # remove false positives and duplicates from detection
        detected_cars = Helper.remove_false_positives(img, bounding_boxes)

        # get detected cars
        # detected_cars_multi_windows = Helper.draw_boxes(img, bounding_boxes, color=(0, 0, 0), thick=3)
        # Visualization.save_detection_multi_windows(img, detected_cars_multi_windows)
        Visualization.save_detection(img, detected_cars)


__main__()
