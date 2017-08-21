import time

import cv2 as cv
import matplotlib.image as mpimg
import numpy as np

from configuration import Configuration
from feature_extraction import FeatureExtraction
from helper import Helper

config = Configuration().__dict__


class Visualization:
    @staticmethod
    def save_detection_multi_windows(img, detected):
        seconds = int(time.time() % 60)
        mpimg.imsave("../buffer/detection-multi-windows/{}-multi-window-detection.png".format(seconds), detected)

    @staticmethod
    def save_detection(img, detected):
        seconds = int(time.time() % 60)
        mpimg.imsave("../buffer/detections/{}-single-window-detection.png".format(seconds), detected)

    @staticmethod
    def save_hog_features(img, hog_image, folder, filename):
        """
        to be used in FeatureExtraction.get_hog_features
        """
        seconds = int(time.time() % 60)
        filename = filename if filename else seconds
        if seconds % 10 == 0:
            mpimg.imsave("{}{}-original.png"
                         .format(folder, filename),
                         img, cmap="gray")
            mpimg.imsave("{}{}-feature.png"
                         .format(folder, filename),
                         hog_image, cmap="gray")

    @staticmethod
    def save_heat_map(img):
        seconds = int(time.time() % 60)
        # if seconds % 10 == 0:
        mpimg.imsave("../buffer/heat-maps/{}-heat-map.png"
                     .format(seconds),
                     img, cmap="gist_heat")

    @staticmethod
    def extract_single_img_features(img):
        """
        combine spatial bin, color histogram and gradient histogram features for a single image
        """
        # Create a list to append feature vectors to
        features = []

        # apply color conversion if other than 'RGB'
        feature_image = Helper.change_cspace(img)

        # get hog features for either specific channel or for all channels
        if config["hog_channel"] == 'ALL':
            hog_features = []
            channels = feature_image.shape[2]
            # get features for all 3 channels
            for channel in range(channels):
                hog_features.append(FeatureExtraction.get_hog_features(feature_image[:, :, channel], feature_vec=True))
                hog_features = np.ravel(hog_features)
        else:
            # get features for specific channel
            hog_features = FeatureExtraction.get_hog_features(feature_image[:, :, config["hog_channel"]],
                                                              feature_vec=True)

        # Apply bin_spatial() to get spatial color features
        bin_features = FeatureExtraction.bin_spatial(feature_image, config["spatial_size"])

        # Apply color_hist() to get color histogram features
        color_hist_features = FeatureExtraction.color_hist(feature_image, config["hist_bins"])

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
            features = Visualization.extract_single_img_features(test_img)
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

    @staticmethod
    def visualize_lanes_fit(img, lanes_img, nonzero, inds, fit):
        """
        visualize the lanes fit
        :param img: source warped binary image
        :param lanes_img: destination warped binary image with lanes drawn
        :param nonzero: nonzero pixels
        :param inds: indices of the nonzero xy pixels
        :param fit: left and right fit
        :return lanes_img: destination warped binary image with lanes drawn
        """
        # Generate x and y values for plotting
        left_lane_inds, right_lane_inds = inds
        nonzero_x, nonzero_y = nonzero

        # Color in left and right line pixels
        lanes_img[nonzero_y[left_lane_inds], nonzero_x[left_lane_inds]] = [255, 0, 0]
        lanes_img[nonzero_y[right_lane_inds], nonzero_x[right_lane_inds]] = [0, 0, 255]

        return lanes_img

    @staticmethod
    def visualize_updated_lanes_fit(img, lanes_img, nonzero, inds, fit):
        """
        visualize updated lanes fit
        :param img: source warped binary image
        :param lanes_img: destination warped binary image with lanes drawn
        :param nonzero: nonzero pixels
        :param inds: indices of the nonzero xy pixels
        :param fit: left and right fit
        :return lanes_img: destination warped binary image with lanes drawn
        """
        # Generate x and y values for plotting
        img_height = img.shape[0]
        left_lane_inds, right_lane_inds = inds
        nonzero_x, nonzero_y = nonzero
        left_fit, right_fit = fit
        margin = 100

        # using Ay^2 + By + C
        plot_y = np.linspace(0, img_height - 1, img_height)
        left_fitx = left_fit[0] * plot_y ** 2 + left_fit[1] * plot_y + left_fit[2]
        right_fitx = right_fit[0] * plot_y ** 2 + right_fit[1] * plot_y + right_fit[2]

        # Create an image to draw on and an image to show the selection window
        window_img = np.zeros_like(lanes_img)

        # Color in left and right line pixels
        lanes_img[nonzero_y[left_lane_inds], nonzero_x[left_lane_inds]] = [255, 0, 0]
        lanes_img[nonzero_y[right_lane_inds], nonzero_x[right_lane_inds]] = [0, 0, 255]

        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, plot_y]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin, plot_y])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))

        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, plot_y]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin, plot_y])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the lane onto the warped blank image
        cv.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
        cv.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))

        lanes_img = cv.addWeighted(lanes_img, 1, window_img, 0.3, 0)

        return lanes_img

    @staticmethod
    def visualize_pipeline(resultant_img, img_dst,
                           binary_image, lane_lines,
                           radius, center_distance,
                           lane_width):
        """
        visualize the important steps of the pipeline
        :param resultant_img: resultant image of the pipeline
        :param img_dst: wrapped binary image
        :param binary_image: binary image
        :param lane_lines: wrapped binary image with lane lines
        :param radius: radius of curvature
        :param center_distance: car distance from center lane
        :param lane_width: width of the lane
        :return: None
        """
        # resize the image for better visualization
        binary_image = cv.cvtColor(binary_image, cv.COLOR_GRAY2BGR)

        # FONT_HERSHEY_SIMPLEX
        font = cv.QT_FONT_NORMAL

        radius_txt = "Radius of Curvature = {}m".format(str(round(radius[0], 3)))
        left_or_right = "left" if center_distance > 0 else "right"
        distance_txt = "Vehicle is {}m {} of center ".format(str(round(abs(center_distance), 2)), left_or_right)

        cv.putText(resultant_img, radius_txt, (10, 30), font, 0.6, (255, 255, 255), 1, cv.LINE_AA)
        cv.putText(resultant_img, distance_txt, (10, 60), font, 0.6, (255, 255, 255), 1, cv.LINE_AA)

        # resize the image for better visualization
        img_dst = cv.resize(img_dst, None, fx=0.4, fy=0.3, interpolation=cv.INTER_LINEAR)
        lane_lines = cv.resize(lane_lines, None, fx=0.4, fy=0.3, interpolation=cv.INTER_LINEAR)

        return resultant_img
