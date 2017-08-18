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

classifier = Classifier.get_trained_classifier(use_pre_trained=True)
hyper_params = Configuration().__dict__
sys.path.append("implementation/")


def __main__():
    video_cap = imageio.get_reader(hyper_params["testing_video"])

    # polynomial lane fit
    lanes_fit = []

    # load calibration parameters:
    camera_matrix, dist_coef = PreProcessing.load_calibration_params()
    for img in video_cap:
        lanes_fit, img = LaneDetection.pipeline(img, lanes_fit, camera_matrix, dist_coef)
        img = cv.resize(img, None, fx=0.6, fy=0.6, interpolation=cv.INTER_LINEAR)

        # testing dataset is in jpg format
        # while training dataset is in png format
        # scaling required
        if hyper_params["is_training_png"]:
            img = Helper.scale_to_png(img)

        # 3 channel without alpha
        img = img[:, :, :3]

        # image dimensions
        width, height = img.shape[1], img.shape[0]

        # region of interest (ROI)
        y_start_top = [int(height / 2), height]

        # get bounding boxes for cars in the image
        bounding_boxes = WindowSearch.get_bounding_boxes(img, classifier, y_start_top)

        # remove false positives and duplicates from detection
        detected_cars = Helper.remove_false_positives(img, bounding_boxes)

        if hyper_params["save_debug_samples"] is True:
            # get detected cars
            detected_cars_multi_windows = Helper.draw_boxes(img, bounding_boxes, color=(0, 0, 0), thick=3)
            Visualization.save_detection_multi_windows(img, detected_cars_multi_windows)
            Visualization.save_detection(img, detected_cars)

        detected_cars = cv.cvtColor(detected_cars, cv.COLOR_BGR2RGB)

        cv.imshow("Detected Cars", detected_cars)
        cv.waitKey(1)


__main__()
