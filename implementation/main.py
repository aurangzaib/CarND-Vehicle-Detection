import glob
import sys

import matplotlib.image as mpimg

from contrib import Contrib

sys.path.append("implementation/")
from classifier import Classifier
from helper import Helper
from window_search import WindowSearch
from configuration import Configuration

hyper_params = Configuration().__dict__

classifier = Classifier.get_trained_classifier(use_pre_trained=True)

imgs = glob.glob(hyper_params["testing"])
for filename in imgs:
    # read image
    img = mpimg.imread(filename)

    # testing dataset is in jpg format
    # while training dataset is in png format
    # scaling required
    img = Helper.scale_to_png(img)

    # 3 channel without alpha
    img = img[:, :, :3]

    # image dimensions
    width, height = img.shape[1], img.shape[0]

    # region of interest (ROI)
    y_start_top = [int(height / 2), height]

    # get bounding boxes for cars in the image
    bounding_boxes = WindowSearch.get_bounding_boxes(img, classifier, y_start_top)

    detected_cars_multi_windows = Helper.draw_boxes(img, bounding_boxes, color=(0, 0, 0), thick=3)
    Contrib.save_detection_multi_windows(detected_cars_multi_windows)

    detected_cars = Helper.remove_false_positives(img, bounding_boxes)
    Contrib.save_detection(detected_cars)

    # plt.pause(0.000001)
