import sys

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
    video_cap = imageio.get_reader(config["testing_video_2"])
    # polynomial lane fit
    lanes_fit = []

    # classifier and scaler
    classifier = Classifier.get_trained_classifier(use_pre_trained=True)

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
            x_start_stop_left, y_start_stop_left = (0, 400), (370, 600)
            bounding_boxes += WindowSearch.get_bounding_boxes(img, classifier,
                                                              x_start_stop_left,
                                                              y_start_stop_left)
            # # get bounding boxes for top side
            x_start_stop_top, y_start_stop_top = (400, 800), (410, 450)
            bounding_boxes += WindowSearch.get_bounding_boxes(img, classifier,
                                                              x_start_stop_top,
                                                              y_start_stop_top)
            # get bounding boxes for right side
            x_start_stop_right, y_start_stop_right = (800, 1270), (370, 600)
            bounding_boxes += WindowSearch.get_bounding_boxes(img, classifier,
                                                              x_start_stop_right,
                                                              y_start_stop_right)
            # remove false positives and duplicates from detection
            detected_cars = Helper.remove_false_positives(img, bounding_boxes)

            # bounding_boxes = np.hstack((bounding_boxes_left, bounding_boxes_top, bounding_boxes_right))
            # get detected cars
            # detected_cars_multi_windows = Helper.draw_boxes(img,
            #                                                 x_start_stop_left, y_start_stop_left,
            #                                                 x_start_stop_right, y_start_stop_right,
            #                                                 x_start_stop_top, y_start_stop_top,
            #                                                 bounding_boxes,
            #                                                 color=(0, 0, 0),
            #                                                 thick=3)

            # Visualization.save_detection_multi_windows(img, detected_cars_multi_windows)
            Visualization.save_detection(img, detected_cars)
            # cv.imshow("result", detected_cars)
            # cv.waitKey(1)


__main__()
