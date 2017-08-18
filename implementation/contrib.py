import time

import cv2 as cv
import matplotlib.image as mpimg
import numpy as np

from configuration import Configuration
from helper import Helper

hyper_params = Configuration().__dict__


class Contrib:
    @staticmethod
    def save_detection_multi_windows(img):
        seconds = int(time.time() % 60)
        if seconds % 10 == 0:
            mpimg.imsave("../buffer/detection-multi-window/detection-multi-window-{}.png"
                         .format(seconds),
                         img)

    @staticmethod
    def save_detection(img):
        seconds = int(time.time() % 60)
        if seconds % 10 == 0:
            mpimg.imsave("../buffer/detection/detection-{}.png"
                         .format(seconds),
                         img)

    @staticmethod
    def save_hog_features(img, hog_image):
        seconds = int(time.time() % 60)
        if seconds % 10 == 0:
            mpimg.imsave("../buffer/hog-features/hog-original-{}.png"
                         .format(seconds),
                         img, cmap="gray")
            mpimg.imsave("../buffer/hog-features/hog-features-{}.png"
                         .format(seconds),
                         hog_image, cmap="gray")

    @staticmethod
    def extract_single_img_features(img):
        """
        combine spatial bin, color histogram and gradient histogram features for a single image
        """
        # Create a list to append feature vectors to
        features = []

        # apply color conversion if other than 'RGB'
        feature_image = Helper.change_cspace(img, hyper_params["cspace"])

        # get hog features for either specific channel or for all channels
        if hyper_params["hog_channel"] == 'ALL':
            hog_features = []
            channels = feature_image.shape[2]
            # get features for all 3 channels
            for channel in range(channels):
                hog_features.append(Helper.get_hog_features(feature_image[:, :, channel],
                                                            feature_vec=True))
                hog_features = np.ravel(hog_features)
        else:
            # get features for specific channel
            hog_features = Helper.get_hog_features(feature_image[:, :, hyper_params["hog_channel"]], feature_vec=True)

        # Apply bin_spatial() to get spatial color features
        bin_features = Helper.bin_spatial(feature_image, hyper_params["spatial_size"])

        # Apply color_hist() to get color histogram features
        color_hist_features = Helper.color_hist(feature_image, hyper_params["hist_bins"])

        # concatenate all 3 types of features
        feature = np.concatenate((bin_features, color_hist_features, hog_features), axis=0)

        # Append the new feature vector to the features list
        features.append(feature)

        # Return list of feature vectors
        return features

    @staticmethod
    def search_windows(img, windows, clf, scaler, ):
        # 1) Create an empty list to receive positive detection windows
        on_windows = []
        # 2) Iterate over all windows in the list
        for window in windows:
            # 3) Extract the test window from original image
            test_img = cv.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
            # 4) Extract features for that window
            features = Contrib.extract_single_img_features(test_img)
            # 5) Scale extracted features to be fed to classifier
            test_features = scaler.transform(np.array(features).reshape(1, -1))
            # 6) Predict using your classifier
            prediction = clf.predict(test_features)
            # 7) If positive (prediction == 1) i.e. car, then save the window
            if prediction == 1:
                on_windows.append(window)
        # 8) Return windows for positive detections
        return on_windows

    @staticmethod
    def get_slide_windows(img,
                          x_start_stop,
                          y_start_stop,
                          xy_window=(64, 64),
                          xy_overlap=(0.5, 0.5)):
        # If x and/or y start/stop positions not defined, set to image size
        img_width, img_height = img.shape[1], img.shape[0]
        if x_start_stop[0] is None:
            x_start_stop[0] = 0
        if x_start_stop[1] is None:
            x_start_stop[1] = img_width
        if y_start_stop[0] is None:
            y_start_stop[0] = 0
        if y_start_stop[1] is None:
            y_start_stop[1] = img_height

        # Compute the span of the region to be searched
        xy_span = (x_start_stop[1] - x_start_stop[0],
                   y_start_stop[1] - y_start_stop[0])

        # Compute the number of pixels per step in x/y
        n_pixels_per_step = (np.int(xy_window[0] * (1 - xy_overlap[0])),
                             np.int(xy_window[1] * (1 - xy_overlap[1])))

        # Compute the number of windows in x/y
        n_buffer = (np.int(xy_window[0] * (xy_overlap[0])),
                    np.int(xy_window[1] * (xy_overlap[1])))

        n_windows = (np.int((xy_span[0] - n_buffer[0]) / n_pixels_per_step[0]),
                     np.int((xy_span[1] - n_buffer[1]) / n_pixels_per_step[1]))

        # Initialize a list to append window positions to
        window_list = []
        # Loop through finding x and y window positions
        for ys in range(n_windows[1]):
            for xs in range(n_windows[0]):
                # Calculate each window position
                xy_start = (xs * n_pixels_per_step[0] + x_start_stop[0],
                            ys * n_pixels_per_step[1] + y_start_stop[0])
                xy_stop = (xy_start[0] + xy_window[0],
                           xy_start[1] + xy_window[1])
                # Append window position to list
                window_list.append(((xy_start[0], xy_start[1]),
                                    (xy_stop[0], xy_stop[1])))
        # Return the list of windows
        return window_list
